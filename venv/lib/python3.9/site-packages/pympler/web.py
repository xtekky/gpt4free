"""
This module provides a web-based memory profiling interface. The Pympler web
frontend exposes process information, tracker statistics, and garbage graphs.
The web frontend uses `Bottle <http://bottlepy.org>`_, a lightweight Python
web framework. Bottle is packaged with Pympler.

The web server can be invoked almost as easily as setting a breakpoint using
*pdb*::

    from pympler.web import start_profiler
    start_profiler()

Calling ``start_profiler`` suspends the current thread and executes the Pympler
web server, exposing profiling data and various facilities of the Pympler
library via a graphic interface.
"""

import sys
import os
import threading

from inspect import getouterframes
from json import dumps
from shutil import rmtree
from tempfile import mkdtemp
from threading import Thread
from weakref import WeakValueDictionary
from wsgiref.simple_server import make_server

from pympler import asizeof
from pympler.garbagegraph import GarbageGraph
from pympler.process import get_current_threads, ProcessMemoryInfo

from pympler.util.stringutils import safe_repr

# Prefer the installed version of bottle.py. If bottle.py is not installed
# fallback to the vendored version.
try:
    import bottle
except ImportError:
    from pympler.util import bottle


class ServerState(threading.local):
    """
    Represents the state of a running server. Needs to be thread local so
    multiple servers can be started in different threads without interfering
    with each other.

    Cache internal structures (garbage graphs, tracker statistics).
    """
    def __init__(self):
        self.server = None
        self.stats = None
        self.garbage_graphs = None
        self.id2ref = WeakValueDictionary()
        self.id2obj = dict()

    def clear_cache(self):
        self.garbage_graphs = None


server = ServerState()


def get_ref(obj):
    """
    Get string reference to object. Stores a weak reference in a dictionary
    using the object's id as the key. If the object cannot be weakly
    referenced (e.g. dictionaries, frame objects), store a strong references
    in a classic dictionary.

    Returns the object's id as a string.
    """
    oid = id(obj)
    try:
        server.id2ref[oid] = obj
    except TypeError:
        server.id2obj[oid] = obj
    return str(oid)


def get_obj(ref):
    """Get object from string reference."""
    oid = int(ref)
    return server.id2ref.get(oid) or server.id2obj[oid]


pympler_path = os.path.dirname(os.path.abspath(__file__))
static_files = os.path.join(pympler_path, 'templates')

bottle.TEMPLATE_PATH.append(static_files)


@bottle.route('/')
@bottle.view('index')
def root():
    """Get overview."""
    pmi = ProcessMemoryInfo()
    return dict(processinfo=pmi)


@bottle.route('/process')
@bottle.view('process')
def process():
    """Get process overview."""
    pmi = ProcessMemoryInfo()
    threads = get_current_threads()
    return dict(info=pmi, threads=threads)


@bottle.route('/tracker')
@bottle.view('tracker')
def tracker_index():
    """Get tracker overview."""
    stats = server.stats
    if stats and stats.snapshots:
        stats.annotate()
        timeseries = []
        for cls in stats.tracked_classes:
            series = []
            for snapshot in stats.snapshots:
                series.append(snapshot.classes.get(cls, {}).get('sum', 0))
            timeseries.append((cls, series))

        series = [s.overhead for s in stats.snapshots]
        timeseries.append(("Profiling overhead", series))

        if stats.snapshots[0].system_total.data_segment:
            # Assume tracked data resides in the data segment
            series = [s.system_total.data_segment - s.tracked_total - s.overhead
                      for s in stats.snapshots]
            timeseries.append(("Data segment", series))
            series = [s.system_total.code_segment for s in stats.snapshots]
            timeseries.append(("Code segment", series))
            series = [s.system_total.stack_segment for s in stats.snapshots]
            timeseries.append(("Stack segment", series))
            series = [s.system_total.shared_segment for s in stats.snapshots]
            timeseries.append(("Shared memory", series))
        else:
            series = [s.total - s.tracked_total - s.overhead
                      for s in stats.snapshots]
            timeseries.append(("Other", series))
        timeseries = [dict(label=label, data=list(enumerate(data)))
                      for label, data in timeseries]
        return dict(snapshots=stats.snapshots, timeseries=dumps(timeseries))
    else:
        return dict(snapshots=[])


@bottle.route('/tracker/class/<clsname>')
@bottle.view('tracker_class')
def tracker_class(clsname):
    """Get class instance details."""
    stats = server.stats
    if not stats:
        bottle.redirect('/tracker')
    stats.annotate()
    return dict(stats=stats, clsname=clsname)


@bottle.route('/refresh')
def refresh():
    """Clear all cached information."""
    server.clear_cache()
    bottle.redirect('/')


@bottle.route('/traceback/<threadid>')
@bottle.view('stacktrace')
def get_traceback(threadid):
    threadid = int(threadid)
    frames = sys._current_frames()
    if threadid in frames:
        frame = frames[threadid]
        stack = getouterframes(frame, 5)
        stack.reverse()
        stack = [(get_ref(f[0].f_locals),) + f[1:] for f in stack]
    else:
        stack = []
    return dict(stack=stack, threadid=threadid)


@bottle.route('/objects/<oid>')
@bottle.view('referents')
def get_obj_referents(oid):
    referents = {}
    obj = get_obj(oid)
    if type(obj) is dict:
        named_objects = asizeof.named_refs(obj)
    else:
        refs = asizeof._getreferents(obj)
        named_objects = [(repr(type(x)), x) for x in refs]
    for name, o in named_objects:
        referents[name] = (get_ref(o), type(o).__name__,
                           safe_repr(o, clip=48), asizeof.asizeof(o))
    return dict(referents=referents)


@bottle.route('/static/<filename>')
def static_file(filename):
    """Get static files (CSS-files)."""
    return bottle.static_file(filename, root=static_files)


def _compute_garbage_graphs():
    """
    Retrieve garbage graph objects from cache, compute if cache is cold.
    """
    if server.garbage_graphs is None:
        server.garbage_graphs = GarbageGraph().split_and_sort()
    return server.garbage_graphs


@bottle.route('/garbage')
@bottle.view('garbage_index')
def garbage_index():
    """Get garbage overview."""
    garbage_graphs = _compute_garbage_graphs()
    return dict(graphs=garbage_graphs)


@bottle.route('/garbage/<index:int>')
@bottle.view('garbage')
def garbage_cycle(index):
    """Get reference cycle details."""
    graph = _compute_garbage_graphs()[int(index)]
    graph.reduce_to_cycles()
    objects = graph.metadata
    objects.sort(key=lambda x: -x.size)
    return dict(objects=objects, index=index)


def _get_graph(graph, filename):
    """Retrieve or render a graph."""
    try:
        rendered = graph.rendered_file
    except AttributeError:
        try:
            graph.render(os.path.join(server.tmpdir, filename), format='png')
            rendered = filename
        except OSError:
            rendered = None
    graph.rendered_file = rendered
    return rendered


@bottle.route('/garbage/graph/<index:int>')
def garbage_graph(index):
    """Get graph representation of reference cycle."""
    graph = _compute_garbage_graphs()[int(index)]
    reduce_graph = bottle.request.GET.get('reduce', '')
    if reduce_graph:
        graph = graph.reduce_to_cycles()
    if not graph:
        return None
    filename = 'garbage%so%s.png' % (index, reduce_graph)
    rendered_file = _get_graph(graph, filename)
    if rendered_file:
        return bottle.static_file(rendered_file, root=server.tmpdir)
    else:
        return None


@bottle.route('/help')
def show_documentation():
    """Redirect to online documentation."""
    bottle.redirect('https://pympler.readthedocs.io/en/latest/')


class PymplerServer(bottle.ServerAdapter):
    """Simple WSGI server."""
    def run(self, handler):
        self.server = make_server(self.host, self.port, handler)
        self.server.serve_forever()


def start_profiler(host='localhost', port=8090, tracker=None, stats=None,
                   debug=False, **kwargs):
    """
    Start the web server to show profiling data. The function suspends the
    Python application (the current thread) until the web server is stopped.

    The only way to stop the server is to signal the running thread, e.g. press
    Ctrl+C in the console. If this isn't feasible for your application use
    `start_in_background` instead.

    During the execution of the web server, profiling data is (lazily) cached
    to improve performance. For example, garbage graphs are rendered when the
    garbage profiling data is requested and are simply retransmitted upon later
    requests.

    The web server can display profiling data from previously taken snapshots
    when `tracker` or `stats` is specified. The former is useful for profiling
    a running application, the latter for off-line analysis. Requires existing
    snapshots taken with
    :py:meth:`~pympler.classtracker.ClassTracker.create_snapshot` or
    :py:meth:`~pympler.classtracker.ClassTracker.start_periodic_snapshots`.

    :param host: the host where the server shall run, default is localhost
    :param port: server listens on the specified port, default is 8090 to allow
        coexistance with common web applications
    :param tracker: `ClassTracker` instance, browse profiling data (on-line
        analysis)
    :param stats: `Stats` instance, analyze `ClassTracker` profiling dumps
        (useful for off-line analysis)
    """
    if tracker and not stats:
        server.stats = tracker.stats
    else:
        server.stats = stats
    try:
        server.tmpdir = mkdtemp(prefix='pympler')
        server.server = PymplerServer(host=host, port=port, **kwargs)
        bottle.debug(debug)
        bottle.run(server=server.server)
    finally:
        rmtree(server.tmpdir)


class ProfilerThread(Thread):
    """Encapsulates a thread to run the web server."""
    def __init__(self, group=None, target=None, name='Pympler web frontend',
                 **kwargs):
        super(ProfilerThread, self).__init__(group=group,
                                             target=target,
                                             name=name)
        self.kwargs = kwargs
        self.daemon = True

    def run(self):
        start_profiler(**self.kwargs)


def start_in_background(**kwargs):
    """
    Start the web server in the background. A new thread is created which
    serves the profiling interface without suspending the current application.

    For the documentation of the parameters see `start_profiler`.

    Returns the created thread object.
    """
    thread = ProfilerThread(**kwargs)
    thread.start()
    return thread
