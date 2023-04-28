"""
Provide saving, loading and presenting gathered `ClassTracker` statistics.
"""

from typing import (
    Any, Dict, IO, Iterable, List, Optional, Tuple, TYPE_CHECKING, Union
)

import os
import pickle
import sys
from copy import deepcopy
from pympler.util.stringutils import trunc, pp, pp_timestamp

from pympler.asizeof import Asized

if TYPE_CHECKING:
    from .classtracker import TrackedObject, ClassTracker, Snapshot


__all__ = ["Stats", "ConsoleStats", "HtmlStats"]


def _ref2key(ref: Asized) -> str:
    return ref.name.split(':')[0]


def _merge_asized(base: Asized, other: Asized, level: int = 0) -> None:
    """
    Merge **Asized** instances `base` and `other` into `base`.
    """
    base.size += other.size
    base.flat += other.flat
    if level > 0:
        base.name = _ref2key(base)
    # Add refs from other to base. Any new refs are appended.
    base.refs = list(base.refs)  # we may need to append items
    refs = {}
    for ref in base.refs:
        refs[_ref2key(ref)] = ref
    for ref in other.refs:
        key = _ref2key(ref)
        if key in refs:
            _merge_asized(refs[key], ref, level=level + 1)
        else:
            # Don't modify existing Asized instances => deepcopy
            base.refs.append(deepcopy(ref))
            base.refs[-1].name = key


def _merge_objects(tref: float, merged: Asized, obj: 'TrackedObject') -> None:
    """
    Merge the snapshot size information of multiple tracked objects.  The
    tracked object `obj` is scanned for size information at time `tref`.
    The sizes are merged into **Asized** instance `merged`.
    """
    size = None
    for (timestamp, tsize) in obj.snapshots:
        if timestamp == tref:
            size = tsize
    if size:
        _merge_asized(merged, size)


def _format_trace(trace: List[Tuple]) -> str:
    """
    Convert the (stripped) stack-trace to a nice readable format. The stack
    trace `trace` is a list of frame records as returned by
    **inspect.stack** but without the frame objects.
    Returns a string.
    """
    lines = []
    for fname, lineno, func, src, _ in trace:
        if src:
            for line in src:
                lines.append('    ' + line.strip() + '\n')
        lines.append('  %s:%4d in %s\n' % (fname, lineno, func))
    return ''.join(lines)


class Stats(object):
    """
    Presents the memory statistics gathered by a `ClassTracker` based on user
    preferences.
    """

    def __init__(self, tracker: 'Optional[ClassTracker]' = None,
                 filename: Optional[str] = None,
                 stream: Optional[IO] = None):
        """
        Initialize the data log structures either from a `ClassTracker`
        instance (argument `tracker`) or a previously dumped file (argument
        `filename`).

        :param tracker: ClassTracker instance
        :param filename: filename of previously dumped statistics
        :param stream: where to print statistics, defaults to ``sys.stdout``
        """
        if stream:
            self.stream = stream
        else:
            self.stream = sys.stdout
        self.tracker = tracker
        self.index = {}  # type: Dict[str, List[TrackedObject]]
        self.snapshots = []  # type: List[Snapshot]
        if tracker:
            self.index = tracker.index
            self.snapshots = tracker.snapshots
            self.history = tracker.history
        self.sorted = []  # type: List[TrackedObject]
        if filename:
            self.load_stats(filename)

    def load_stats(self, fdump: Union[str, IO[bytes]]) -> None:
        """
        Load the data from a dump file.
        The argument `fdump` can be either a filename or an open file object
        that requires read access.
        """
        if isinstance(fdump, str):
            fdump = open(fdump, 'rb')
        self.index = pickle.load(fdump)
        self.snapshots = pickle.load(fdump)
        self.sorted = []

    def dump_stats(self, fdump: Union[str, IO[bytes]], close: bool = True
                   ) -> None:
        """
        Dump the logged data to a file.
        The argument `file` can be either a filename or an open file object
        that requires write access. `close` controls if the file is closed
        before leaving this method (the default behaviour).
        """
        if self.tracker:
            self.tracker.stop_periodic_snapshots()

        if isinstance(fdump, str):
            fdump = open(fdump, 'wb')
        pickle.dump(self.index, fdump, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.snapshots, fdump, protocol=pickle.HIGHEST_PROTOCOL)
        if close:
            fdump.close()

    def _init_sort(self) -> None:
        """
        Prepare the data to be sorted.
        If not yet sorted, import all tracked objects from the tracked index.
        Extend the tracking information by implicit information to make
        sorting easier (DSU pattern).
        """
        if not self.sorted:
            # Identify the snapshot that tracked the largest amount of memory.
            tmax = None
            maxsize = 0
            for snapshot in self.snapshots:
                if snapshot.tracked_total > maxsize:
                    tmax = snapshot.timestamp
            for key in list(self.index.keys()):
                for tobj in self.index[key]:
                    tobj.classname = key  # type: ignore
                    tobj.size = tobj.get_max_size()  # type: ignore
                    tobj.tsize = tobj.get_size_at_time(tmax)  # type: ignore
                self.sorted.extend(self.index[key])

    def sort_stats(self, *args: str) -> 'Stats':
        """
        Sort the tracked objects according to the supplied criteria. The
        argument is a string identifying the basis of a sort (example: 'size'
        or 'classname'). When more than one key is provided, then additional
        keys are used as secondary criteria when there is equality in all keys
        selected before them. For example, ``sort_stats('name', 'size')`` will
        sort all the entries according to their class name, and resolve all
        ties (identical class names) by sorting by size.  The criteria are
        fields in the tracked object instances. Results are stored in the
        ``self.sorted`` list which is used by ``Stats.print_stats()`` and other
        methods. The fields available for sorting are:

            'classname'
                the name with which the class was registered
            'name'
                the classname
            'birth'
                creation timestamp
            'death'
                destruction timestamp
            'size'
                the maximum measured size of the object
            'tsize'
                the measured size during the largest snapshot
            'repr'
                string representation of the object

        Note that sorts on size are in descending order (placing most memory
        consuming items first), whereas name, repr, and creation time searches
        are in ascending order (alphabetical).

        The function returns self to allow calling functions on the result::

            stats.sort_stats('size').reverse_order().print_stats()
        """

        criteria = ('classname', 'tsize', 'birth', 'death',
                    'name', 'repr', 'size')

        if not set(criteria).issuperset(set(args)):
            raise ValueError("Invalid sort criteria")

        if not args:
            args = criteria

        def args_to_tuple(obj: 'TrackedObject') -> Tuple[str, ...]:
            keys: List[str] = []
            for attr in args:
                attribute = getattr(obj, attr, '')
                if attr in ('tsize', 'size'):
                    attribute = -int(attribute)
                keys.append(attribute)
            return tuple(keys)

        self._init_sort()
        self.sorted.sort(key=args_to_tuple)

        return self

    def reverse_order(self) -> 'Stats':
        """
        Reverse the order of the tracked instance index `self.sorted`.
        """
        self._init_sort()
        self.sorted.reverse()
        return self

    def annotate(self) -> None:
        """
        Annotate all snapshots with class-based summaries.
        """
        for snapshot in self.snapshots:
            self.annotate_snapshot(snapshot)

    def annotate_snapshot(self, snapshot: 'Snapshot'
                          ) -> Dict[str, Dict[str, Any]]:
        """
        Store additional statistical data in snapshot.
        """
        if snapshot.classes is not None:
            return snapshot.classes

        snapshot.classes = {}

        for classname in list(self.index.keys()):
            total = 0
            active = 0
            merged = Asized(0, 0)
            for tobj in self.index[classname]:
                _merge_objects(snapshot.timestamp, merged, tobj)
                total += tobj.get_size_at_time(snapshot.timestamp)
                if (tobj.birth < snapshot.timestamp and
                        (tobj.death is None or
                         tobj.death > snapshot.timestamp)):
                    active += 1

            try:
                pct = total * 100.0 / snapshot.total
            except ZeroDivisionError:  # pragma: no cover
                pct = 0
            try:
                avg = total / active
            except ZeroDivisionError:
                avg = 0

            snapshot.classes[classname] = dict(sum=total,
                                               avg=avg,
                                               pct=pct,
                                               active=active)
            snapshot.classes[classname]['merged'] = merged

        return snapshot.classes

    @property
    def tracked_classes(self) -> List[str]:
        """Return a list of all tracked classes occurring in any snapshot."""
        return sorted(list(self.index.keys()))


class ConsoleStats(Stats):
    """
    Presentation layer for `Stats` to be used in text-based consoles.
    """

    def _print_refs(self, refs: Iterable[Asized], total: int,
                    prefix: str = '    ', level: int = 1, minsize: int = 0,
                    minpct: float = 0.1) -> None:
        """
        Print individual referents recursively.
        """
        lrefs = list(refs)
        lrefs.sort(key=lambda x: x.size)
        lrefs.reverse()
        for ref in lrefs:
            if ref.size > minsize and (ref.size * 100.0 / total) > minpct:
                self.stream.write('%-50s %-14s %3d%% [%d]\n' % (
                    trunc(prefix + str(ref.name), 50),
                    pp(ref.size),
                    int(ref.size * 100.0 / total),
                    level
                ))
                self._print_refs(ref.refs, total, prefix=prefix + '  ',
                                 level=level + 1)

    def print_object(self, tobj: 'TrackedObject') -> None:
        """
        Print the gathered information of object `tobj` in human-readable
        format.
        """
        if tobj.death:
            self.stream.write('%-32s ( free )   %-35s\n' % (
                trunc(tobj.name, 32, left=True), trunc(tobj.repr, 35)))
        else:
            self.stream.write('%-32s 0x%08x %-35s\n' % (
                trunc(tobj.name, 32, left=True),

                tobj.id,
                trunc(tobj.repr, 35)
            ))
        if tobj.trace:
            self.stream.write(_format_trace(tobj.trace))
        for (timestamp, size) in tobj.snapshots:
            self.stream.write('  %-30s %s\n' % (
                pp_timestamp(timestamp), pp(size.size)
            ))
            self._print_refs(size.refs, size.size)
        if tobj.death is not None:
            self.stream.write('  %-30s finalize\n' % (
                pp_timestamp(tobj.death),
            ))

    def print_stats(self, clsname: Optional[str] = None, limit: float = 1.0
                    ) -> None:
        """
        Write tracked objects to stdout.  The output can be filtered and
        pruned.  Only objects are printed whose classname contain the substring
        supplied by the `clsname` argument.  The output can be pruned by
        passing a `limit` value.

        :param clsname: Only print objects whose classname contain the given
            substring.
        :param limit: If `limit` is a float smaller than one, only the supplied
            percentage of the total tracked data is printed. If `limit` is
            bigger than one, this number of tracked objects are printed.
            Tracked objects are first filtered, and then pruned (if specified).
        """
        if self.tracker:
            self.tracker.stop_periodic_snapshots()

        if not self.sorted:
            self.sort_stats()

        _sorted = self.sorted

        if clsname:
            _sorted = [
                to for to in _sorted
                if clsname in to.classname  # type: ignore
            ]

        if limit < 1.0:
            limit = max(1, int(len(self.sorted) * limit))
        _sorted = _sorted[:int(limit)]

        # Emit per-instance data
        for tobj in _sorted:
            self.print_object(tobj)

    def print_summary(self) -> None:
        """
        Print per-class summary for each snapshot.
        """
        # Emit class summaries for each snapshot
        classlist = self.tracked_classes

        fobj = self.stream

        fobj.write('---- SUMMARY ' + '-' * 66 + '\n')
        for snapshot in self.snapshots:
            classes = self.annotate_snapshot(snapshot)
            fobj.write('%-35s %11s %12s %12s %5s\n' % (
                trunc(snapshot.desc, 35),
                'active',
                pp(snapshot.asizeof_total),
                'average',
                'pct'
            ))
            for classname in classlist:
                info = classes[classname]
                fobj.write('  %-33s %11d %12s %12s %4d%%\n' % (
                    trunc(classname, 33),
                    info['active'],
                    pp(info['sum']),
                    pp(info['avg']),
                    info['pct']
                ))
        fobj.write('-' * 79 + '\n')


class HtmlStats(Stats):
    """
    Output the `ClassTracker` statistics as HTML pages and graphs.
    """

    style = """<style type="text/css">
        table { width:100%; border:1px solid #000; border-spacing:0px; }
        td, th { border:0px; }
        div { width:200px; padding:10px; background-color:#FFEECC; }
        #nb { border:0px; }
        #tl { margin-top:5mm; margin-bottom:5mm; }
        #p1 { padding-left: 5px; }
        #p2 { padding-left: 50px; }
        #p3 { padding-left: 100px; }
        #p4 { padding-left: 150px; }
        #p5 { padding-left: 200px; }
        #p6 { padding-left: 210px; }
        #p7 { padding-left: 220px; }
        #hl { background-color:#FFFFCC; }
        #r1 { background-color:#BBBBBB; }
        #r2 { background-color:#CCCCCC; }
        #r3 { background-color:#DDDDDD; }
        #r4 { background-color:#EEEEEE; }
        #r5,#r6,#r7 { background-color:#FFFFFF; }
        #num { text-align:right; }
    </style>
    """

    nopylab_msg = """<div color="#FFCCCC">Could not generate %s chart!
    Install <a href="http://matplotlib.sourceforge.net/">Matplotlib</a>
    to generate charts.</div>\n"""

    chart_tag = '<img src="%s">\n'
    header = "<html><head><title>%s</title>%s</head><body>\n"
    tableheader = '<table border="1">\n'
    tablefooter = '</table>\n'
    footer = '</body></html>\n'

    refrow = """<tr id="r%(level)d">
        <td id="p%(level)d">%(name)s</td>
        <td id="num">%(size)s</td>
        <td id="num">%(pct)3.1f%%</td></tr>"""

    def _print_refs(self, fobj: IO, refs: Iterable[Asized], total: int,
                    level: int = 1, minsize: int = 0, minpct: float = 0.1
                    ) -> None:
        """
        Print individual referents recursively.
        """
        lrefs = list(refs)
        lrefs.sort(key=lambda x: x.size)
        lrefs.reverse()
        if level == 1:
            fobj.write('<table>\n')
        for ref in lrefs:
            if ref.size > minsize and (ref.size * 100.0 / total) > minpct:
                data = dict(level=level,
                            name=trunc(str(ref.name), 128),
                            size=pp(ref.size),
                            pct=ref.size * 100.0 / total)
                fobj.write(self.refrow % data)
                self._print_refs(fobj, ref.refs, total, level=level + 1)
        if level == 1:
            fobj.write("</table>\n")

    class_summary = """<p>%(cnt)d instances of %(cls)s were registered. The
        average size is %(avg)s, the minimal size is %(min)s, the maximum size
        is %(max)s.</p>\n"""
    class_snapshot = '''<h3>Snapshot: %(name)s, %(total)s occupied by instances
        of class %(cls)s</h3>\n'''

    def print_class_details(self, fname: str, classname: str) -> None:
        """
        Print detailed statistics and instances for the class `classname`. All
        data will be written to the file `fname`.
        """
        fobj = open(fname, "w")
        fobj.write(self.header % (classname, self.style))

        fobj.write("<h1>%s</h1>\n" % (classname))

        sizes = [tobj.get_max_size() for tobj in self.index[classname]]
        total = 0
        for s in sizes:
            total += s
        data = {'cnt': len(self.index[classname]), 'cls': classname}
        data['avg'] = pp(total / len(sizes))
        data['max'] = pp(max(sizes))
        data['min'] = pp(min(sizes))
        fobj.write(self.class_summary % data)

        fobj.write(self.charts[classname])

        fobj.write("<h2>Coalesced Referents per Snapshot</h2>\n")
        for snapshot in self.snapshots:
            if snapshot.classes and classname in snapshot.classes:
                merged = snapshot.classes[classname]['merged']
                fobj.write(self.class_snapshot % {
                    'name': snapshot.desc,
                    'cls': classname,
                    'total': pp(merged.size),
                })
                if merged.refs:
                    self._print_refs(fobj, merged.refs, merged.size)
                else:
                    fobj.write('<p>No per-referent sizes recorded.</p>\n')

        fobj.write("<h2>Instances</h2>\n")
        for tobj in self.index[classname]:
            fobj.write('<table id="tl" width="100%" rules="rows">\n')
            fobj.write('<tr><td id="hl" width="140px">Instance</td>' +
                       '<td id="hl">%s at 0x%08x</td></tr>\n' %
                       (tobj.name, tobj.id))
            if tobj.repr:
                fobj.write("<tr><td>Representation</td>" +
                           "<td>%s&nbsp;</td></tr>\n" % tobj.repr)
            fobj.write("<tr><td>Lifetime</td><td>%s - %s</td></tr>\n" %
                       (pp_timestamp(tobj.birth), pp_timestamp(tobj.death)))
            if tobj.trace:
                trace = "<pre>%s</pre>" % (_format_trace(tobj.trace))
                fobj.write("<tr><td>Instantiation</td><td>%s</td></tr>\n" %
                           trace)
            for (timestamp, size) in tobj.snapshots:
                fobj.write("<tr><td>%s</td>" % pp_timestamp(timestamp))
                if not size.refs:
                    fobj.write("<td>%s</td></tr>\n" % pp(size.size))
                else:
                    fobj.write("<td>%s" % pp(size.size))
                    self._print_refs(fobj, size.refs, size.size)
                    fobj.write("</td></tr>\n")
            fobj.write("</table>\n")

        fobj.write(self.footer)
        fobj.close()

    snapshot_cls_header = """<tr>
        <th id="hl">Class</th>
        <th id="hl" align="right">Instance #</th>
        <th id="hl" align="right">Total</th>
        <th id="hl" align="right">Average size</th>
        <th id="hl" align="right">Share</th></tr>\n"""

    snapshot_cls = """<tr>
        <td>%(cls)s</td>
        <td align="right">%(active)d</td>
        <td align="right">%(sum)s</td>
        <td align="right">%(avg)s</td>
        <td align="right">%(pct)3.2f%%</td></tr>\n"""

    snapshot_summary = """<p>Total virtual memory assigned to the program
        at that time was %(sys)s, which includes %(overhead)s profiling
        overhead. The ClassTracker tracked %(tracked)s in total. The measurable
        objects including code objects but excluding overhead have a total size
        of %(asizeof)s.</p>\n"""

    def relative_path(self, filepath: str, basepath: Optional[str] = None
                      ) -> str:
        """
        Convert the filepath path to a relative path against basepath. By
        default basepath is self.basedir.
        """
        if basepath is None:
            basepath = self.basedir
        if not basepath:
            return filepath
        if filepath.startswith(basepath):
            filepath = filepath[len(basepath):]
        if filepath and filepath[0] == os.sep:
            filepath = filepath[1:]
        return filepath

    def create_title_page(self, filename: str, title: str = '') -> None:
        """
        Output the title page.
        """
        fobj = open(filename, "w")
        fobj.write(self.header % (title, self.style))

        fobj.write("<h1>%s</h1>\n" % title)
        fobj.write("<h2>Memory distribution over time</h2>\n")
        fobj.write(self.charts['snapshots'])

        fobj.write("<h2>Snapshots statistics</h2>\n")
        fobj.write('<table id="nb">\n')

        classlist = list(self.index.keys())
        classlist.sort()

        for snapshot in self.snapshots:
            fobj.write('<tr><td>\n')
            fobj.write('<table id="tl" rules="rows">\n')
            fobj.write("<h3>%s snapshot at %s</h3>\n" % (
                snapshot.desc or 'Untitled',
                pp_timestamp(snapshot.timestamp)
            ))

            data = {}
            data['sys'] = pp(snapshot.system_total.vsz)
            data['tracked'] = pp(snapshot.tracked_total)
            data['asizeof'] = pp(snapshot.asizeof_total)
            data['overhead'] = pp(getattr(snapshot, 'overhead', 0))

            fobj.write(self.snapshot_summary % data)

            if snapshot.tracked_total:
                fobj.write(self.snapshot_cls_header)
                for classname in classlist:
                    if snapshot.classes:
                        info = snapshot.classes[classname].copy()
                        path = self.relative_path(self.links[classname])
                        info['cls'] = '<a href="%s">%s</a>' % (path, classname)
                        info['sum'] = pp(info['sum'])
                        info['avg'] = pp(info['avg'])
                        fobj.write(self.snapshot_cls % info)
            fobj.write('</table>')
            fobj.write('</td><td>\n')
            if snapshot.tracked_total:
                fobj.write(self.charts[snapshot])
            fobj.write('</td></tr>\n')

        fobj.write("</table>\n")
        fobj.write(self.footer)
        fobj.close()

    def create_lifetime_chart(self, classname: str, filename: str = '') -> str:
        """
        Create chart that depicts the lifetime of the instance registered with
        `classname`. The output is written to `filename`.
        """
        try:
            from pylab import figure, title, xlabel, ylabel, plot, savefig
        except ImportError:
            return HtmlStats.nopylab_msg % (classname + " lifetime")

        cnt = []
        for tobj in self.index[classname]:
            cnt.append([tobj.birth, 1])
            if tobj.death:
                cnt.append([tobj.death, -1])
        cnt.sort()
        for i in range(1, len(cnt)):
            cnt[i][1] += cnt[i - 1][1]

        x = [t for [t, c] in cnt]
        y = [c for [t, c] in cnt]

        figure()
        xlabel("Execution time [s]")
        ylabel("Instance #")
        title("%s instances" % classname)
        plot(x, y, 'o')
        savefig(filename)

        return self.chart_tag % (os.path.basename(filename))

    def create_snapshot_chart(self, filename: str = '') -> str:
        """
        Create chart that depicts the memory allocation over time apportioned
        to the tracked classes.
        """
        try:
            from pylab import (figure, title, xlabel, ylabel, plot, fill,
                               legend, savefig)
            import matplotlib.mlab as mlab
        except ImportError:
            return self.nopylab_msg % ("memory allocation")

        classlist = self.tracked_classes

        times = [snapshot.timestamp for snapshot in self.snapshots]
        base = [0.0] * len(self.snapshots)
        poly_labels = []
        polys = []
        for cn in classlist:
            pct = [snapshot.classes[cn]['pct'] for snapshot in self.snapshots
                   if snapshot.classes is not None]
            if pct and max(pct) > 3.0:
                sz = [float(fp.classes[cn]['sum']) / (1024 * 1024)
                      for fp in self.snapshots
                      if fp.classes is not None]
                sz = [sx + sy for sx, sy in zip(base, sz)]
                xp, yp = mlab.poly_between(times, base, sz)
                polys.append(((xp, yp), {'label': cn}))
                poly_labels.append(cn)
                base = sz

        figure()
        title("Snapshot Memory")
        xlabel("Execution Time [s]")
        ylabel("Virtual Memory [MiB]")

        sizes = [float(fp.asizeof_total) / (1024 * 1024)
                 for fp in self.snapshots]
        plot(times, sizes, 'r--', label='Total')
        sizes = [float(fp.tracked_total) / (1024 * 1024)
                 for fp in self.snapshots]
        plot(times, sizes, 'b--', label='Tracked total')

        for (args, kwds) in polys:
            fill(*args, **kwds)
        legend(loc=2)
        savefig(filename)

        return self.chart_tag % (self.relative_path(filename))

    def create_pie_chart(self, snapshot: 'Snapshot', filename: str = '') -> str:
        """
        Create a pie chart that depicts the distribution of the allocated
        memory for a given `snapshot`. The chart is saved to `filename`.
        """
        try:
            from pylab import figure, title, pie, axes, savefig
            from pylab import sum as pylab_sum
        except ImportError:
            return self.nopylab_msg % ("pie_chart")

        # Don't bother illustrating a pie without pieces.
        if not snapshot.tracked_total or snapshot.classes is None:
            return ''

        classlist = []
        sizelist = []
        for k, v in list(snapshot.classes.items()):
            if v['pct'] > 3.0:
                classlist.append(k)
                sizelist.append(v['sum'])
        sizelist.insert(0, snapshot.asizeof_total - pylab_sum(sizelist))
        classlist.insert(0, 'Other')

        title("Snapshot (%s) Memory Distribution" % (snapshot.desc))
        figure(figsize=(8, 8))
        axes([0.1, 0.1, 0.8, 0.8])
        pie(sizelist, labels=classlist)
        savefig(filename, dpi=50)

        return self.chart_tag % (self.relative_path(filename))

    def create_html(self, fname: str, title: str = "ClassTracker Statistics"
                    ) -> None:
        """
        Create HTML page `fname` and additional files in a directory derived
        from `fname`.
        """
        # Create a folder to store the charts and additional HTML files.
        self.basedir = os.path.dirname(os.path.abspath(fname))
        self.filesdir = os.path.splitext(fname)[0] + '_files'
        if not os.path.isdir(self.filesdir):
            os.mkdir(self.filesdir)
        self.filesdir = os.path.abspath(self.filesdir)
        self.links = {}  # type: Dict[str, str]

        # Annotate all snapshots in advance
        self.annotate()

        # Create charts. The tags to show the images are returned and stored in
        # the self.charts dictionary. This allows to return alternative text if
        # the chart creation framework is not available.
        self.charts = {}  # type: Dict[Union[str, Snapshot], str]
        fn = os.path.join(self.filesdir, 'timespace.png')
        self.charts['snapshots'] = self.create_snapshot_chart(fn)

        for fp, idx in zip(self.snapshots, list(range(len(self.snapshots)))):
            fn = os.path.join(self.filesdir, 'fp%d.png' % (idx))
            self.charts[fp] = self.create_pie_chart(fp, fn)

        for cn in list(self.index.keys()):
            fn = os.path.join(self.filesdir, cn.replace('.', '_') + '-lt.png')
            self.charts[cn] = self.create_lifetime_chart(cn, fn)

        # Create HTML pages first for each class and then the index page.
        for cn in list(self.index.keys()):
            fn = os.path.join(self.filesdir, cn.replace('.', '_') + '.html')
            self.links[cn] = fn
            self.print_class_details(fn, cn)

        self.create_title_page(fname, title=title)
