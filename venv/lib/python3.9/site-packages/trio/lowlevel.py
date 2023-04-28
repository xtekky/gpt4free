"""
This namespace represents low-level functionality not intended for daily use,
but useful for extending Trio's functionality.
"""

import select as _select
import sys
import typing as _t

# This is the union of a subset of trio/_core/ and some things from trio/*.py.
# See comments in trio/__init__.py for details. To make static analysis easier,
# this lists all possible symbols from trio._core, and then we prune those that
# aren't available on this system. After that we add some symbols from trio/*.py.

# Generally available symbols
from ._core import (
    cancel_shielded_checkpoint,
    Abort,
    wait_task_rescheduled,
    enable_ki_protection,
    disable_ki_protection,
    currently_ki_protected,
    Task,
    checkpoint,
    current_task,
    ParkingLot,
    UnboundedQueue,
    RunVar,
    TrioToken,
    current_trio_token,
    temporarily_detach_coroutine_object,
    permanently_detach_coroutine_object,
    reattach_detached_coroutine_object,
    current_statistics,
    reschedule,
    remove_instrument,
    add_instrument,
    current_clock,
    current_root_task,
    checkpoint_if_cancelled,
    spawn_system_task,
    wait_readable,
    wait_writable,
    notify_closing,
    start_thread_soon,
    start_guest_run,
)

from ._subprocess import open_process

if sys.platform == "win32":
    # Windows symbols
    from ._core import (
        current_iocp,
        register_with_iocp,
        wait_overlapped,
        monitor_completion_key,
        readinto_overlapped,
        write_overlapped,
    )
    from ._wait_for_object import WaitForSingleObject
else:
    # Unix symbols
    from ._unix_pipes import FdStream

    # Kqueue-specific symbols
    if sys.platform != "linux" and (_t.TYPE_CHECKING or not hasattr(_select, "epoll")):
        from ._core import (
            current_kqueue,
            monitor_kevent,
            wait_kevent,
        )

del sys
