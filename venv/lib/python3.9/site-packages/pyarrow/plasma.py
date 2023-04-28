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


import contextlib
import os
import pyarrow as pa
import shutil
import subprocess
import sys
import tempfile
import time
import warnings

from pyarrow._plasma import (ObjectID, ObjectNotAvailable,  # noqa
                             PlasmaBuffer, PlasmaClient, connect,
                             PlasmaObjectExists, PlasmaObjectNotFound,
                             PlasmaStoreFull)


# The Plasma TensorFlow Operator needs to be compiled on the end user's
# machine since the TensorFlow ABI is not stable between versions.
# The following code checks if the operator is already present. If not,
# the function build_plasma_tensorflow_op can be used to compile it.


TF_PLASMA_OP_PATH = os.path.join(pa.__path__[0], "tensorflow", "plasma_op.so")


tf_plasma_op = None


def load_plasma_tensorflow_op():
    global tf_plasma_op
    import tensorflow as tf
    tf_plasma_op = tf.load_op_library(TF_PLASMA_OP_PATH)


def build_plasma_tensorflow_op():
    global tf_plasma_op
    try:
        import tensorflow as tf
        print("TensorFlow version: " + tf.__version__)
    except ImportError:
        pass
    else:
        print("Compiling Plasma TensorFlow Op...")
        dir_path = os.path.dirname(os.path.realpath(__file__))
        cc_path = os.path.join(dir_path, "tensorflow", "plasma_op.cc")
        so_path = os.path.join(dir_path, "tensorflow", "plasma_op.so")
        tf_cflags = tf.sysconfig.get_compile_flags()
        if sys.platform == 'darwin':
            tf_cflags = ["-undefined", "dynamic_lookup"] + tf_cflags
        cmd = ["g++", "-std=c++17", "-g", "-shared", cc_path,
               "-o", so_path, "-DNDEBUG", "-I" + pa.get_include()]
        cmd += ["-L" + dir for dir in pa.get_library_dirs()]
        cmd += ["-lplasma", "-larrow_python", "-larrow", "-fPIC"]
        cmd += tf_cflags
        cmd += tf.sysconfig.get_link_flags()
        cmd += ["-O2"]
        if tf.test.is_built_with_cuda():
            cmd += ["-DGOOGLE_CUDA"]
        print("Running command " + str(cmd))
        subprocess.check_call(cmd)
        tf_plasma_op = tf.load_op_library(TF_PLASMA_OP_PATH)


@contextlib.contextmanager
def start_plasma_store(plasma_store_memory,
                       use_valgrind=False, use_profiler=False,
                       plasma_directory=None, use_hugepages=False,
                       external_store=None):
    """
    DEPRECATED: Start a plasma store process.

    .. deprecated:: 10.0.0
       Plasma is deprecated since Arrow 10.0.0. It will be removed
       in 12.0.0 or so.

    Parameters
    ----------
    plasma_store_memory : int
        Capacity of the plasma store in bytes.
    use_valgrind : bool
        True if the plasma store should be started inside of valgrind. If this
        is True, use_profiler must be False.
    use_profiler : bool
        True if the plasma store should be started inside a profiler. If this
        is True, use_valgrind must be False.
    plasma_directory : str
        Directory where plasma memory mapped files will be stored.
    use_hugepages : bool
        True if the plasma store should use huge pages.
    external_store : str
        External store to use for evicted objects.

    Yields
    -------
    plasma_store_name : str
        Name of the plasma store socket
    proc : subprocess.Popen
        Process ID of the plasma store process
    """
    warnings.warn(
        "Plasma is deprecated since Arrow 10.0.0. It will be removed in "
        "12.0.0 or so.",
        DeprecationWarning)

    if use_valgrind and use_profiler:
        raise Exception("Cannot use valgrind and profiler at the same time.")

    tmpdir = tempfile.mkdtemp(prefix='test_plasma-')
    try:
        plasma_store_name = os.path.join(tmpdir, 'plasma.sock')
        plasma_store_executable = os.path.join(
            pa.__path__[0], "plasma-store-server")
        if not os.path.exists(plasma_store_executable):
            # Fallback to sys.prefix/bin/ (conda)
            plasma_store_executable = os.path.join(
                sys.prefix, "bin", "plasma-store-server")
        command = [plasma_store_executable,
                   "-s", plasma_store_name,
                   "-m", str(plasma_store_memory)]
        if plasma_directory:
            command += ["-d", plasma_directory]
        if use_hugepages:
            command += ["-h"]
        if external_store is not None:
            command += ["-e", external_store]
        stdout_file = None
        stderr_file = None
        if use_valgrind:
            command = ["valgrind",
                       "--track-origins=yes",
                       "--leak-check=full",
                       "--show-leak-kinds=all",
                       "--leak-check-heuristics=stdstring",
                       "--error-exitcode=1"] + command
            proc = subprocess.Popen(command, stdout=stdout_file,
                                    stderr=stderr_file)
            time.sleep(1.0)
        elif use_profiler:
            command = ["valgrind", "--tool=callgrind"] + command
            proc = subprocess.Popen(command, stdout=stdout_file,
                                    stderr=stderr_file)
            time.sleep(1.0)
        else:
            proc = subprocess.Popen(command, stdout=stdout_file,
                                    stderr=stderr_file)
            time.sleep(0.1)
        rc = proc.poll()
        if rc is not None:
            raise RuntimeError("plasma_store exited unexpectedly with "
                               "code %d" % (rc,))

        yield plasma_store_name, proc
    finally:
        if proc.poll() is None:
            proc.kill()
        shutil.rmtree(tmpdir)
