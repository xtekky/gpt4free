"""The tracker module allows you to track changes in the memory usage over
time.

Using the SummaryTracker, you can create summaries and compare them
with each other. Stored summaries can be ignored during comparison,
avoiding the observer effect.

The ObjectTracker allows to monitor object creation. You create objects from
one time and compare with objects from an earlier time.

"""
import gc
import inspect

from pympler import muppy, summary
from pympler.util import compat


class SummaryTracker(object):
    """ Helper class to track changes between two summaries taken.

    Detailed information on single objects will be lost, e.g. object size or
    object id. But often summaries are sufficient to monitor the memory usage
    over the lifetime of an application.

    On initialisation, a first summary is taken. Every time `diff` is called,
    a new summary will be created. Thus, a diff between the new and the last
    summary can be extracted.

    Be aware that filtering out previous summaries is time-intensive. You
    should therefore restrict yourself to the number of summaries you really
    need.

    """
    def __init__(self, ignore_self=True):
        """Constructor.

        The number of summaries managed by the tracker has a performance
        impact on new summaries, iff you decide to exclude them from further
        summaries. Therefore it is suggested to use them economically.

        Keyword arguments:
        ignore_self -- summaries managed by this object will be ignored.
        """
        self.s0 = summary.summarize(muppy.get_objects())
        self.summaries = {}
        self.ignore_self = ignore_self

    def create_summary(self):
        """Return a summary.

        See also the notes on ignore_self in the class as well as the
        initializer documentation.

        """
        if not self.ignore_self:
            res = summary.summarize(muppy.get_objects())
        else:
            # If the user requested the data required to store summaries to be
            # ignored in the summaries, we need to identify all objects which
            # are related to each summary stored.
            # Thus we build a list of all objects used for summary storage as
            # well as a dictionary which tells us how often an object is
            # referenced by the summaries.
            # During this identification process, more objects are referenced,
            # namely int objects identifying referenced objects as well as the
            # corresponding count.
            # For all these objects it will be checked whether they are
            # referenced from outside the monitor's scope. If not, they will be
            # subtracted from the snapshot summary, otherwise they are
            # included (as this indicates that they are relevant to the
            # application).

            all_of_them = []  # every single object
            ref_counter = {}  # how often it is referenced; (id(o), o) pairs

            def store_info(o):
                all_of_them.append(o)
                if id(o) in ref_counter:
                    ref_counter[id(o)] += 1
                else:
                    ref_counter[id(o)] = 1

            # store infos on every single object related to the summaries
            store_info(self.summaries)
            for k, v in self.summaries.items():
                store_info(k)
                summary._traverse(v, store_info)

            # do the summary
            res = summary.summarize(muppy.get_objects())

            # remove ids stored in the ref_counter
            for _id in ref_counter:
                # referenced in frame, ref_counter, ref_counter.keys()
                if len(gc.get_referrers(_id)) == (3):
                    summary._subtract(res, _id)
            for o in all_of_them:
                # referenced in frame, summary, all_of_them
                if len(gc.get_referrers(o)) == (ref_counter[id(o)] + 2):
                    summary._subtract(res, o)

        return res

    def diff(self, summary1=None, summary2=None):
        """Compute diff between to summaries.

        If no summary is provided, the diff from the last to the current
        summary is used. If summary1 is provided the diff from summary1
        to the current summary is used. If summary1 and summary2 are
        provided, the diff between these two is used.

        """
        res = None
        if summary2 is None:
            self.s1 = self.create_summary()
            if summary1 is None:
                res = summary.get_diff(self.s0, self.s1)
            else:
                res = summary.get_diff(summary1, self.s1)
            self.s0 = self.s1
        else:
            if summary1 is not None:
                res = summary.get_diff(summary1, summary2)
            else:
                raise ValueError(
                    "You cannot provide summary2 without summary1.")
        return summary._sweep(res)

    def print_diff(self, summary1=None, summary2=None):
        """Compute diff between to summaries and print it.

        If no summary is provided, the diff from the last to the current
        summary is used. If summary1 is provided the diff from summary1
        to the current summary is used. If summary1 and summary2 are
        provided, the diff between these two is used.
        """
        summary.print_(self.diff(summary1=summary1, summary2=summary2))

    def format_diff(self, summary1=None, summary2=None):
        """Compute diff between to summaries and return a list of formatted
        lines.

        If no summary is provided, the diff from the last to the current
        summary is used. If summary1 is provided the diff from summary1
        to the current summary is used. If summary1 and summary2 are
        provided, the diff between these two is used.
        """
        return summary.format_(self.diff(summary1=summary1, summary2=summary2))

    def store_summary(self, key):
        """Store a current summary in self.summaries."""
        self.summaries[key] = self.create_summary()


class ObjectTracker(object):
    """
    Helper class to track changes in the set of existing objects.

    Each time you invoke a diff with this tracker, the objects which existed
    during the last invocation are compared with the objects which exist during
    the current invocation.

    Please note that in order to do so, strong references to all objects will
    be stored. This means that none of these objects can be garbage collected.
    A use case for the ObjectTracker is the monitoring of a state which should
    be stable, but you see new objects being created nevertheless. With the
    ObjectTracker you can identify these new objects.

    """

    # Some precaution needs to be taken when handling frame objects (see
    # warning at http://docs.python.org/lib/inspect-stack.html). All ignore
    # lists used need to be emptied so no frame objects remain referenced.

    def __init__(self):
        """On initialisation, the current state of objects is stored.

        Note that all objects which exist at this point in time will not be
        released until you destroy this ObjectTracker instance.
        """
        self.o0 = self._get_objects(ignore=(inspect.currentframe(),))

    def _get_objects(self, ignore=()):
        """Get all currently existing objects.

        XXX - ToDo: This method is a copy&paste from muppy.get_objects, but
        some modifications are applied. Specifically, it allows to ignore
        objects (which includes the current frame).

        keyword arguments
        ignore -- list of objects to ignore
        """
        def remove_ignore(objects, ignore=()):
            # remove all objects listed in the ignore list
            res = []
            for o in objects:
                if not compat.object_in_list(o, ignore):
                    res.append(o)
            return res

        tmp = gc.get_objects()
        ignore += (inspect.currentframe(), self, ignore, remove_ignore)
        if hasattr(self, 'o0'):
            ignore += (self.o0,)
        if hasattr(self, 'o1'):
            ignore += (self.o1,)
        # this implies that referenced objects are also ignored
        tmp = remove_ignore(tmp, ignore)
        res = []
        for o in tmp:
            # gc.get_objects returns only container objects, but we also want
            # the objects referenced by them
            refs = muppy.get_referents(o)
            for ref in refs:
                if not gc.is_tracked(ref):
                    # we already got the container objects, now we only add
                    # non-container objects
                    res.append(ref)
        res.extend(tmp)
        res = muppy._remove_duplicates(res)
        if ignore is not None:
            # repeat to filter out objects which may have been referenced
            res = remove_ignore(res, ignore)
        # manual cleanup, see comment above
        del ignore
        return res

    def get_diff(self, ignore=()):
        """Get the diff to the last time the  state of objects was measured.

        keyword arguments
        ignore -- list of objects to ignore
        """
        # ignore this and the caller frame
        self.o1 = self._get_objects(ignore+(inspect.currentframe(),))
        diff = muppy.get_diff(self.o0, self.o1)
        self.o0 = self.o1
        # manual cleanup, see comment above
        return diff

    def print_diff(self, ignore=()):
        """Print the diff to the last time the state of objects was measured.

        keyword arguments
        ignore -- list of objects to ignore
        """
        # ignore this and the caller frame
        for line in self.format_diff(ignore+(inspect.currentframe(),)):
            print(line)

    def format_diff(self, ignore=()):
        """Format the diff to the last time the state of objects was measured.

        keyword arguments
        ignore -- list of objects to ignore
        """
        # ignore this and the caller frame
        lines = []
        diff = self.get_diff(ignore+(inspect.currentframe(),))
        lines.append("Added objects:")
        for line in summary.format_(summary.summarize(diff['+'])):
            lines.append(line)
        lines.append("Removed objects:")
        for line in summary.format_(summary.summarize(diff['-'])):
            lines.append(line)
        return lines
