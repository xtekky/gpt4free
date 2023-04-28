# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import numpy as np
import pytest


def run_tensorflow_test_with_dtype(tf, plasma, plasma_store_name,
                                   client, use_gpu, dtype):
    FORCE_DEVICE = '/gpu' if use_gpu else '/cpu'

    object_id = np.random.bytes(20)

    data = np.random.randn(3, 244, 244).astype(dtype)
    ones = np.ones((3, 244, 244)).astype(dtype)

    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=True))

    def ToPlasma():
        data_tensor = tf.constant(data)
        ones_tensor = tf.constant(ones)
        return plasma.tf_plasma_op.tensor_to_plasma(
            [data_tensor, ones_tensor],
            object_id,
            plasma_store_socket_name=plasma_store_name)

    def FromPlasma():
        return plasma.tf_plasma_op.plasma_to_tensor(
            object_id,
            dtype=tf.as_dtype(dtype),
            plasma_store_socket_name=plasma_store_name)

    with tf.device(FORCE_DEVICE):
        to_plasma = ToPlasma()
        from_plasma = FromPlasma()

    z = from_plasma + 1

    sess.run(to_plasma)
    # NOTE(zongheng): currently it returns a flat 1D tensor.
    # So reshape manually.
    out = sess.run(from_plasma)

    out = np.split(out, 2)
    out0 = out[0].reshape(3, 244, 244)
    out1 = out[1].reshape(3, 244, 244)

    sess.run(z)

    assert np.array_equal(data, out0), "Data not equal!"
    assert np.array_equal(ones, out1), "Data not equal!"

    # Try getting the data from Python
    plasma_object_id = plasma.ObjectID(object_id)
    obj = client.get(plasma_object_id)

    # Deserialized Tensor should be 64-byte aligned.
    assert obj.ctypes.data % 64 == 0

    result = np.split(obj, 2)
    result0 = result[0].reshape(3, 244, 244)
    result1 = result[1].reshape(3, 244, 244)

    assert np.array_equal(data, result0), "Data not equal!"
    assert np.array_equal(ones, result1), "Data not equal!"


@pytest.mark.plasma
@pytest.mark.tensorflow
@pytest.mark.skip(reason='Until ARROW-4259 is resolved')
def test_plasma_tf_op(use_gpu=False):
    import pyarrow.plasma as plasma
    import tensorflow as tf

    plasma.build_plasma_tensorflow_op()

    if plasma.tf_plasma_op is None:
        pytest.skip("TensorFlow Op not found")

    with plasma.start_plasma_store(10**8) as (plasma_store_name, p):
        client = plasma.connect(plasma_store_name)
        for dtype in [np.float32, np.float64,
                      np.int8, np.int16, np.int32, np.int64]:
            run_tensorflow_test_with_dtype(tf, plasma, plasma_store_name,
                                           client, use_gpu, dtype)

        # Make sure the objects have been released.
        for _, info in client.list().items():
            assert info['ref_count'] == 0
