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

// This header needs to be included multiple times.

#ifdef _WIN32

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

// The Windows API defines macros from *File resolving to either
// *FileA or *FileW.  Need to undo them.
#ifdef CopyFile
#undef CopyFile
#endif
#ifdef CreateFile
#undef CreateFile
#endif
#ifdef DeleteFile
#undef DeleteFile
#endif

// Other annoying Windows macro definitions...
#ifdef IN
#undef IN
#endif
#ifdef OUT
#undef OUT
#endif

// Note that we can't undefine OPTIONAL, because it can be used in other
// Windows headers...

#endif  // _WIN32
