"""
Memory usage profiler for Python.

"""
import inspect
import sys

from pympler import muppy


class MProfiler(object):
    """A memory usage profiler class.

    Memory data for each function is stored as a 3-element list in the
    dictionary self.memories. The index is always a codepoint (see below).
    The following are the definitions of the members:

    [0] = The number of times this function was called
    [1] = Minimum memory consumption when this function was measured.
    [2] = Maximum memory consumption when this function was measured.

    A codepoint is a list of 3-tuple of the type
    (filename, functionname, linenumber). You can omit either element, which
    will cause the profiling to be triggered if any of the other criteria
    match. E.g.
    - (None, foo, None), will profile any foo function,
    - (bar, foo, None) will profile only the foo function from the bar file,
    - (bar, foo, 17) will profile only line 17 of the foo function defined
      in the file bar.

    Additionally, you can define on what events you want the profiling be
    triggered. Possible events are defined in
    http://docs.python.org/lib/debugger-hooks.html.

    If you do not define either codepoints or events, the profiler will
    record the memory usage in at every codepoint and event.

    """

    def __init__(self, codepoints=None, events=None):
        """
        keyword arguments:
        codepoints -- a list of points in code to monitor (defaults to all
            codepoints)
        events -- a list of events to monitor (defaults to all events)
        """
        self.memories = {}
        self.codepoints = codepoints
        self.events = events

    def codepoint_included(self, codepoint):
        """Check if codepoint matches any of the defined codepoints."""
        if self.codepoints is None:
            return True
        for cp in self.codepoints:
            mismatch = False
            for i in range(len(cp)):
                if (cp[i] is not None) and (cp[i] != codepoint[i]):
                    mismatch = True
                    break
            if not mismatch:
                return True
        return False

    def profile(self, frame, event, arg):  # arg req to match signature
        """Profiling method used to profile matching codepoints and events."""
        if (self.events is None) or (event in self.events):
            frame_info = inspect.getframeinfo(frame)
            cp = (frame_info[0], frame_info[2], frame_info[1])
            if self.codepoint_included(cp):
                objects = muppy.get_objects()
                size = muppy.get_size(objects)
                if cp not in self.memories:
                    self.memories[cp] = [0, 0, 0, 0]
                    self.memories[cp][0] = 1
                    self.memories[cp][1] = size
                    self.memories[cp][2] = size
                else:
                    self.memories[cp][0] += 1
                    if self.memories[cp][1] > size:
                        self.memories[cp][1] = size
                    if self.memories[cp][2] < size:
                        self.memories[cp][2] = size

    def run(self, cmd):
        sys.setprofile(self.profile)
        try:
            exec(cmd)
        finally:
            sys.setprofile(None)
        return self


if __name__ == "__main__":
    p = MProfiler()
    p.run("print('hello')")
    print(p.memories)
