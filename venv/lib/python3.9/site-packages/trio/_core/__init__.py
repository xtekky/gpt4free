"""
This namespace represents the core functionality that has to be built-in
and deal with private internal data structures. Things in this namespace
are publicly available in either trio, trio.lowlevel, or trio.testing.
"""

import sys

from ._exceptions import (
    TrioInternalError,
    RunFinishedError,
    WouldBlock,
    Cancelled,
    BusyResourceError,
    ClosedResourceError,
    BrokenResourceError,
    EndOfChannel,
)

from ._ki import (
    enable_ki_protection,
    disable_ki_protection,
    currently_ki_protected,
)

# Imports that always exist
from ._run import (
    Task,
    CancelScope,
    run,
    open_nursery,
    checkpoint,
    current_task,
    current_effective_deadline,
    checkpoint_if_cancelled,
    TASK_STATUS_IGNORED,
    current_statistics,
    current_trio_token,
    reschedule,
    remove_instrument,
    add_instrument,
    current_clock,
    current_root_task,
    spawn_system_task,
    current_time,
    wait_all_tasks_blocked,
    wait_readable,
    wait_writable,
    notify_closing,
    Nursery,
    start_guest_run,
)

# Has to come after _run to resolve a circular import
from ._traps import (
    cancel_shielded_checkpoint,
    Abort,
    wait_task_rescheduled,
    temporarily_detach_coroutine_object,
    permanently_detach_coroutine_object,
    reattach_detached_coroutine_object,
)

from ._entry_queue import TrioToken

from ._parking_lot import ParkingLot

from ._unbounded_queue import UnboundedQueue

from ._local import RunVar

from ._thread_cache import start_thread_soon

from ._mock_clock import MockClock

# Windows imports
if sys.platform == "win32":
    from ._run import (
        monitor_completion_key,
        current_iocp,
        register_with_iocp,
        wait_overlapped,
        write_overlapped,
        readinto_overlapped,
    )
# Kqueue imports
elif sys.platform != "linux" and sys.platform != "win32":
    from ._run import current_kqueue, monitor_kevent, wait_kevent

del sys  # It would be better to import sys as _sys, but mypy does not understand it
