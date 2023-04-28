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

#ifndef _WIN32
#define ARROW_HAVE_SIGACTION 1
#endif

#include <atomic>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if ARROW_HAVE_SIGACTION
#include <signal.h>  // Needed for struct sigaction
#endif

#include "arrow/status.h"
#include "arrow/type_fwd.h"
#include "arrow/util/macros.h"
#include "arrow/util/windows_fixup.h"

namespace arrow {
namespace internal {

// NOTE: 8-bit path strings on Windows are encoded using UTF-8.
// Using MBCS would fail encoding some paths.

#if defined(_WIN32)
using NativePathString = std::wstring;
#else
using NativePathString = std::string;
#endif

class ARROW_EXPORT PlatformFilename {
 public:
  struct Impl;

  ~PlatformFilename();
  PlatformFilename();
  PlatformFilename(const PlatformFilename&);
  PlatformFilename(PlatformFilename&&);
  PlatformFilename& operator=(const PlatformFilename&);
  PlatformFilename& operator=(PlatformFilename&&);
  explicit PlatformFilename(NativePathString path);
  explicit PlatformFilename(const NativePathString::value_type* path);

  const NativePathString& ToNative() const;
  std::string ToString() const;

  PlatformFilename Parent() const;
  Result<PlatformFilename> Real() const;

  // These functions can fail for character encoding reasons.
  static Result<PlatformFilename> FromString(const std::string& file_name);
  Result<PlatformFilename> Join(const std::string& child_name) const;

  PlatformFilename Join(const PlatformFilename& child_name) const;

  bool operator==(const PlatformFilename& other) const;
  bool operator!=(const PlatformFilename& other) const;

  // Made public to avoid the proliferation of friend declarations.
  const Impl* impl() const { return impl_.get(); }

 private:
  std::unique_ptr<Impl> impl_;

  explicit PlatformFilename(Impl impl);
};

/// Create a directory if it doesn't exist.
///
/// Return whether the directory was created.
ARROW_EXPORT
Result<bool> CreateDir(const PlatformFilename& dir_path);

/// Create a directory and its parents if it doesn't exist.
///
/// Return whether the directory was created.
ARROW_EXPORT
Result<bool> CreateDirTree(const PlatformFilename& dir_path);

/// Delete a directory's contents (but not the directory itself) if it exists.
///
/// Return whether the directory existed.
ARROW_EXPORT
Result<bool> DeleteDirContents(const PlatformFilename& dir_path,
                               bool allow_not_found = true);

/// Delete a directory tree if it exists.
///
/// Return whether the directory existed.
ARROW_EXPORT
Result<bool> DeleteDirTree(const PlatformFilename& dir_path, bool allow_not_found = true);

// Non-recursively list the contents of the given directory.
// The returned names are the children's base names, not including dir_path.
ARROW_EXPORT
Result<std::vector<PlatformFilename>> ListDir(const PlatformFilename& dir_path);

/// Delete a file if it exists.
///
/// Return whether the file existed.
ARROW_EXPORT
Result<bool> DeleteFile(const PlatformFilename& file_path, bool allow_not_found = true);

/// Return whether a file exists.
ARROW_EXPORT
Result<bool> FileExists(const PlatformFilename& path);

// TODO expose this more publicly to make it available from io/file.h?
/// A RAII wrapper for a file descriptor.
///
/// The underlying file descriptor is automatically closed on destruction.
/// Moving is supported with well-defined semantics.
/// Furthermore, closing is idempotent.
class ARROW_EXPORT FileDescriptor {
 public:
  FileDescriptor() = default;
  explicit FileDescriptor(int fd) : fd_(fd) {}
  FileDescriptor(FileDescriptor&&);
  FileDescriptor& operator=(FileDescriptor&&);

  ~FileDescriptor();

  Status Close();

  /// May return -1 if closed or default-initialized
  int fd() const { return fd_.load(); }

  /// Detach and return the underlying file descriptor
  int Detach();

  bool closed() const { return fd_.load() == -1; }

 protected:
  static void CloseFromDestructor(int fd);

  std::atomic<int> fd_{-1};
};

/// Open a file for reading and return a file descriptor.
ARROW_EXPORT
Result<FileDescriptor> FileOpenReadable(const PlatformFilename& file_name);

/// Open a file for writing and return a file descriptor.
ARROW_EXPORT
Result<FileDescriptor> FileOpenWritable(const PlatformFilename& file_name,
                                        bool write_only = true, bool truncate = true,
                                        bool append = false);

/// Read from current file position.  Return number of bytes read.
ARROW_EXPORT
Result<int64_t> FileRead(int fd, uint8_t* buffer, int64_t nbytes);
/// Read from given file position.  Return number of bytes read.
ARROW_EXPORT
Result<int64_t> FileReadAt(int fd, uint8_t* buffer, int64_t position, int64_t nbytes);

ARROW_EXPORT
Status FileWrite(int fd, const uint8_t* buffer, const int64_t nbytes);
ARROW_EXPORT
Status FileTruncate(int fd, const int64_t size);

ARROW_EXPORT
Status FileSeek(int fd, int64_t pos);
ARROW_EXPORT
Status FileSeek(int fd, int64_t pos, int whence);
ARROW_EXPORT
Result<int64_t> FileTell(int fd);
ARROW_EXPORT
Result<int64_t> FileGetSize(int fd);

ARROW_EXPORT
Status FileClose(int fd);

struct Pipe {
  FileDescriptor rfd;
  FileDescriptor wfd;

  Status Close() { return rfd.Close() & wfd.Close(); }
};

ARROW_EXPORT
Result<Pipe> CreatePipe();

ARROW_EXPORT
Status SetPipeFileDescriptorNonBlocking(int fd);

class ARROW_EXPORT SelfPipe {
 public:
  static Result<std::shared_ptr<SelfPipe>> Make(bool signal_safe);
  virtual ~SelfPipe();

  /// \brief Wait for a wakeup.
  ///
  /// Status::Invalid is returned if the pipe has been shutdown.
  /// Otherwise the next sent payload is returned.
  virtual Result<uint64_t> Wait() = 0;

  /// \brief Wake up the pipe by sending a payload.
  ///
  /// This method is async-signal-safe if `signal_safe` was set to true.
  virtual void Send(uint64_t payload) = 0;

  /// \brief Wake up the pipe and shut it down.
  virtual Status Shutdown() = 0;
};

ARROW_EXPORT
int64_t GetPageSize();

struct MemoryRegion {
  void* addr;
  size_t size;
};

ARROW_EXPORT
Status MemoryMapRemap(void* addr, size_t old_size, size_t new_size, int fildes,
                      void** new_addr);
ARROW_EXPORT
Status MemoryAdviseWillNeed(const std::vector<MemoryRegion>& regions);

ARROW_EXPORT
Result<std::string> GetEnvVar(const char* name);
ARROW_EXPORT
Result<std::string> GetEnvVar(const std::string& name);
ARROW_EXPORT
Result<NativePathString> GetEnvVarNative(const char* name);
ARROW_EXPORT
Result<NativePathString> GetEnvVarNative(const std::string& name);

ARROW_EXPORT
Status SetEnvVar(const char* name, const char* value);
ARROW_EXPORT
Status SetEnvVar(const std::string& name, const std::string& value);
ARROW_EXPORT
Status DelEnvVar(const char* name);
ARROW_EXPORT
Status DelEnvVar(const std::string& name);

ARROW_EXPORT
std::string ErrnoMessage(int errnum);
#if _WIN32
ARROW_EXPORT
std::string WinErrorMessage(int errnum);
#endif

ARROW_EXPORT
std::shared_ptr<StatusDetail> StatusDetailFromErrno(int errnum);
#if _WIN32
ARROW_EXPORT
std::shared_ptr<StatusDetail> StatusDetailFromWinError(int errnum);
#endif
ARROW_EXPORT
std::shared_ptr<StatusDetail> StatusDetailFromSignal(int signum);

template <typename... Args>
Status StatusFromErrno(int errnum, StatusCode code, Args&&... args) {
  return Status::FromDetailAndArgs(code, StatusDetailFromErrno(errnum),
                                   std::forward<Args>(args)...);
}

template <typename... Args>
Status IOErrorFromErrno(int errnum, Args&&... args) {
  return StatusFromErrno(errnum, StatusCode::IOError, std::forward<Args>(args)...);
}

#if _WIN32
template <typename... Args>
Status StatusFromWinError(int errnum, StatusCode code, Args&&... args) {
  return Status::FromDetailAndArgs(code, StatusDetailFromWinError(errnum),
                                   std::forward<Args>(args)...);
}

template <typename... Args>
Status IOErrorFromWinError(int errnum, Args&&... args) {
  return StatusFromWinError(errnum, StatusCode::IOError, std::forward<Args>(args)...);
}
#endif

template <typename... Args>
Status StatusFromSignal(int signum, StatusCode code, Args&&... args) {
  return Status::FromDetailAndArgs(code, StatusDetailFromSignal(signum),
                                   std::forward<Args>(args)...);
}

template <typename... Args>
Status CancelledFromSignal(int signum, Args&&... args) {
  return StatusFromSignal(signum, StatusCode::Cancelled, std::forward<Args>(args)...);
}

ARROW_EXPORT
int ErrnoFromStatus(const Status&);

// Always returns 0 on non-Windows platforms (for Python).
ARROW_EXPORT
int WinErrorFromStatus(const Status&);

ARROW_EXPORT
int SignalFromStatus(const Status&);

class ARROW_EXPORT TemporaryDir {
 public:
  ~TemporaryDir();

  /// '/'-terminated path to the temporary dir
  const PlatformFilename& path() { return path_; }

  /// Create a temporary subdirectory in the system temporary dir,
  /// named starting with `prefix`.
  static Result<std::unique_ptr<TemporaryDir>> Make(const std::string& prefix);

 private:
  PlatformFilename path_;

  explicit TemporaryDir(PlatformFilename&&);
};

class ARROW_EXPORT SignalHandler {
 public:
  typedef void (*Callback)(int);

  SignalHandler();
  explicit SignalHandler(Callback cb);
#if ARROW_HAVE_SIGACTION
  explicit SignalHandler(const struct sigaction& sa);
#endif

  Callback callback() const;
#if ARROW_HAVE_SIGACTION
  const struct sigaction& action() const;
#endif

 protected:
#if ARROW_HAVE_SIGACTION
  // Storing the full sigaction allows to restore the entire signal handling
  // configuration.
  struct sigaction sa_;
#else
  Callback cb_;
#endif
};

/// \brief Return the current handler for the given signal number.
ARROW_EXPORT
Result<SignalHandler> GetSignalHandler(int signum);

/// \brief Set a new handler for the given signal number.
///
/// The old signal handler is returned.
ARROW_EXPORT
Result<SignalHandler> SetSignalHandler(int signum, const SignalHandler& handler);

/// \brief Reinstate the signal handler
///
/// For use in signal handlers.  This is needed on platforms without sigaction()
/// such as Windows, as the default signal handler is restored there as
/// soon as a signal is raised.
ARROW_EXPORT
void ReinstateSignalHandler(int signum, SignalHandler::Callback handler);

/// \brief Send a signal to the current process
///
/// The thread which will receive the signal is unspecified.
ARROW_EXPORT
Status SendSignal(int signum);

/// \brief Send a signal to the given thread
///
/// This function isn't supported on Windows.
ARROW_EXPORT
Status SendSignalToThread(int signum, uint64_t thread_id);

/// \brief Get an unpredictable random seed
///
/// This function may be slightly costly, so should only be used to initialize
/// a PRNG, not to generate a large amount of random numbers.
/// It is better to use this function rather than std::random_device, unless
/// absolutely necessary (e.g. to generate a cryptographic secret).
ARROW_EXPORT
int64_t GetRandomSeed();

/// \brief Get the current thread id
///
/// In addition to having the same properties as std::thread, the returned value
/// is a regular integer value, which is more convenient than an opaque type.
ARROW_EXPORT
uint64_t GetThreadId();

/// \brief Get the current memory used by the current process in bytes
///
/// This function supports Windows, Linux, and Mac and will return 0 otherwise
ARROW_EXPORT
int64_t GetCurrentRSS();

/// \brief Get the total memory available to the system in bytes
///
/// This function supports Windows, Linux, and Mac and will return 0 otherwise
ARROW_EXPORT
int64_t GetTotalMemoryBytes();

}  // namespace internal
}  // namespace arrow
