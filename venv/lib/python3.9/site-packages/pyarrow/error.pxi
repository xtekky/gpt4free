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

from cpython.exc cimport PyErr_CheckSignals, PyErr_SetInterrupt

from pyarrow.includes.libarrow cimport CStatus
from pyarrow.includes.libarrow_python cimport IsPyError, RestorePyError
from pyarrow.includes.common cimport c_string

from contextlib import contextmanager
import os
import signal
import threading

from pyarrow.util import _break_traceback_cycle_from_frame


class ArrowException(Exception):
    pass


class ArrowInvalid(ValueError, ArrowException):
    pass


class ArrowMemoryError(MemoryError, ArrowException):
    pass


class ArrowKeyError(KeyError, ArrowException):
    def __str__(self):
        # Override KeyError.__str__, as it uses the repr() of the key
        return ArrowException.__str__(self)


class ArrowTypeError(TypeError, ArrowException):
    pass


class ArrowNotImplementedError(NotImplementedError, ArrowException):
    pass


class ArrowCapacityError(ArrowException):
    pass


class ArrowIndexError(IndexError, ArrowException):
    pass


class ArrowSerializationError(ArrowException):
    pass


class ArrowCancelled(ArrowException):
    def __init__(self, message, signum=None):
        super().__init__(message)
        self.signum = signum


# Compatibility alias
ArrowIOError = IOError


# This function could be written directly in C++ if we didn't
# define Arrow-specific subclasses (ArrowInvalid etc.)
cdef int check_status(const CStatus& status) nogil except -1:
    if status.ok():
        return 0

    with gil:
        if IsPyError(status):
            RestorePyError(status)
            return -1

        # We don't use Status::ToString() as it would redundantly include
        # the C++ class name.
        message = frombytes(status.message(), safe=True)
        detail = status.detail()
        if detail != nullptr:
            message += ". Detail: " + frombytes(detail.get().ToString(),
                                                safe=True)

        if status.IsInvalid():
            raise ArrowInvalid(message)
        elif status.IsIOError():
            # Note: OSError constructor is
            #   OSError(message)
            # or
            #   OSError(errno, message, filename=None)
            # or (on Windows)
            #   OSError(errno, message, filename, winerror)
            errno = ErrnoFromStatus(status)
            winerror = WinErrorFromStatus(status)
            if winerror != 0:
                raise IOError(errno, message, None, winerror)
            elif errno != 0:
                raise IOError(errno, message)
            else:
                raise IOError(message)
        elif status.IsOutOfMemory():
            raise ArrowMemoryError(message)
        elif status.IsKeyError():
            raise ArrowKeyError(message)
        elif status.IsNotImplemented():
            raise ArrowNotImplementedError(message)
        elif status.IsTypeError():
            raise ArrowTypeError(message)
        elif status.IsCapacityError():
            raise ArrowCapacityError(message)
        elif status.IsIndexError():
            raise ArrowIndexError(message)
        elif status.IsSerializationError():
            raise ArrowSerializationError(message)
        elif status.IsCancelled():
            signum = SignalFromStatus(status)
            if signum > 0:
                raise ArrowCancelled(message, signum)
            else:
                raise ArrowCancelled(message)
        else:
            message = frombytes(status.ToString(), safe=True)
            raise ArrowException(message)


# This is an API function for C++ PyArrow
cdef api int pyarrow_internal_check_status(const CStatus& status) \
        nogil except -1:
    return check_status(status)


cdef class StopToken:
    cdef void init(self, CStopToken stop_token):
        self.stop_token = move(stop_token)


cdef c_bool signal_handlers_enabled = True


def enable_signal_handlers(c_bool enable):
    """
    Enable or disable interruption of long-running operations.

    By default, certain long running operations will detect user
    interruptions, such as by pressing Ctrl-C.  This detection relies
    on setting a signal handler for the duration of the long-running
    operation, and may therefore interfere with other frameworks or
    libraries (such as an event loop).

    Parameters
    ----------
    enable : bool
        Whether to enable user interruption by setting a temporary
        signal handler.
    """
    global signal_handlers_enabled
    signal_handlers_enabled = enable


# For internal use

# Whether we need a workaround for https://bugs.python.org/issue42248
have_signal_refcycle = (sys.version_info < (3, 8, 10) or
                        (3, 9) <= sys.version_info < (3, 9, 5) or
                        sys.version_info[:2] == (3, 10))

cdef class SignalStopHandler:
    cdef:
        StopToken _stop_token
        vector[int] _signals
        c_bool _enabled

    def __cinit__(self):
        self._enabled = False

        self._init_signals()
        if have_signal_refcycle:
            _break_traceback_cycle_from_frame(sys._getframe(0))

        self._stop_token = StopToken()

        if not self._signals.empty():
            maybe_source = SetSignalStopSource()
            if not maybe_source.ok():
                # See ARROW-11841 / ARROW-17173: in complex interaction
                # scenarios (such as R calling into Python), SetSignalStopSource()
                # may have already activated a signal-receiving StopSource.
                # Just warn instead of erroring out.
                maybe_source.status().Warn()
            else:
                self._stop_token.init(deref(maybe_source).token())
                self._enabled = True

    def _init_signals(self):
        if (signal_handlers_enabled and
                threading.current_thread() is threading.main_thread()):
            self._signals = [
                sig for sig in (signal.SIGINT, signal.SIGTERM)
                if signal.getsignal(sig) not in (signal.SIG_DFL,
                                                 signal.SIG_IGN, None)]

    def __enter__(self):
        if self._enabled:
            check_status(RegisterCancellingSignalHandler(self._signals))
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self._enabled:
            UnregisterCancellingSignalHandler()
        if exc_value is None:
            # Make sure we didn't lose a signal
            try:
                check_status(self._stop_token.stop_token.Poll())
            except ArrowCancelled as e:
                exc_value = e
        if isinstance(exc_value, ArrowCancelled):
            if exc_value.signum:
                # Re-emit the exact same signal. We restored the Python signal
                # handler above, so it should receive it.
                if os.name == 'nt':
                    SendSignal(exc_value.signum)
                else:
                    SendSignalToThread(exc_value.signum,
                                       threading.main_thread().ident)
            else:
                # Simulate Python receiving a SIGINT
                # (see https://bugs.python.org/issue43356 for why we can't
                #  simulate the exact signal number)
                PyErr_SetInterrupt()
            # Maximize chances of the Python signal handler being executed now.
            # Otherwise a potential KeyboardInterrupt might be missed by an
            # immediately enclosing try/except block.
            PyErr_CheckSignals()
            # ArrowCancelled will be re-raised if PyErr_CheckSignals()
            # returned successfully.

    def __dealloc__(self):
        if self._enabled:
            ResetSignalStopSource()

    @property
    def stop_token(self):
        return self._stop_token
