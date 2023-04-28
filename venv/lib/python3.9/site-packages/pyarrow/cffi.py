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

from __future__ import absolute_import

import cffi

c_source = """
    struct ArrowSchema {
      // Array type description
      const char* format;
      const char* name;
      const char* metadata;
      int64_t flags;
      int64_t n_children;
      struct ArrowSchema** children;
      struct ArrowSchema* dictionary;

      // Release callback
      void (*release)(struct ArrowSchema*);
      // Opaque producer-specific data
      void* private_data;
    };

    struct ArrowArray {
      // Array data description
      int64_t length;
      int64_t null_count;
      int64_t offset;
      int64_t n_buffers;
      int64_t n_children;
      const void** buffers;
      struct ArrowArray** children;
      struct ArrowArray* dictionary;

      // Release callback
      void (*release)(struct ArrowArray*);
      // Opaque producer-specific data
      void* private_data;
    };

    struct ArrowArrayStream {
      int (*get_schema)(struct ArrowArrayStream*, struct ArrowSchema* out);
      int (*get_next)(struct ArrowArrayStream*, struct ArrowArray* out);

      const char* (*get_last_error)(struct ArrowArrayStream*);

      // Release callback
      void (*release)(struct ArrowArrayStream*);
      // Opaque producer-specific data
      void* private_data;
    };
    """

# TODO use out-of-line mode for faster import and avoid C parsing
ffi = cffi.FFI()
ffi.cdef(c_source)
