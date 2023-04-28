// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#pragma once

// Workaround for multithreading on XCode 9, see
// https://issues.apache.org/jira/browse/ARROW-1622 and
// https://github.com/tensorflow/tensorflow/issues/13220#issuecomment-331579775
// This should be a short-term fix until the problem is fixed upstream.
#ifdef __APPLE__
#ifndef _MACH_PORT_T
#define _MACH_PORT_T
#include <sys/_types.h> /* __darwin_mach_port_t */
typedef __darwin_mach_port_t mach_port_t;
#include <pthread.h>
mach_port_t pthread_mach_thread_np(pthread_t);
#endif /* _MACH_PORT_T */
#endif /* __APPLE__ */
