
from pympler.refgraph import ReferenceGraph
from pympler.util.stringutils import trunc, pp

import sys
import gc

__all__ = ['GarbageGraph', 'start_debug_garbage', 'end_debug_garbage']


class GarbageGraph(ReferenceGraph):
    """
    The ``GarbageGraph`` is a ``ReferenceGraph`` that illustrates the objects
    building reference cycles. The garbage collector is switched to debug mode
    (all identified garbage is stored in `gc.garbage`) and the garbage
    collector is invoked. The collected objects are then illustrated in a
    directed graph.

    Large graphs can be reduced to the actual cycles by passing ``reduce=True``
    to the constructor.

    It is recommended to disable the garbage collector when using the
    ``GarbageGraph``.

    >>> from pympler.garbagegraph import GarbageGraph, start_debug_garbage
    >>> start_debug_garbage()
    >>> l = []
    >>> l.append(l)
    >>> del l
    >>> gb = GarbageGraph()
    >>> gb.render('garbage.eps')
    True
    """
    def __init__(self, reduce=False, collectable=True):
        """
        Initialize the GarbageGraph with the objects identified by the garbage
        collector. If `collectable` is true, every reference cycle is recorded.
        Otherwise only uncollectable objects are reported.
        """
        if collectable:
            gc.set_debug(gc.DEBUG_SAVEALL)
        else:
            gc.set_debug(0)
        gc.collect()

        ReferenceGraph.__init__(self, gc.garbage, reduce)

    def print_stats(self, stream=None):
        """
        Log annotated garbage objects to console or file.

        :param stream: open file, uses sys.stdout if not given
        """
        if not stream:  # pragma: no cover
            stream = sys.stdout
        self.metadata.sort(key=lambda x: -x.size)
        stream.write('%-10s %8s %-12s %-46s\n' % ('id', 'size', 'type',
                                                  'representation'))
        for g in self.metadata:
            stream.write('0x%08x %8d %-12s %-46s\n' % (g.id, g.size,
                                                       trunc(g.type, 12),
                                                       trunc(g.str, 46)))
        stream.write('Garbage: %8d collected objects (%s in cycles): %12s\n' %
                     (self.count, self.num_in_cycles, pp(self.total_size)))


def start_debug_garbage():
    """
    Turn off garbage collector to analyze *collectable* reference cycles.
    """
    gc.collect()
    gc.disable()


def end_debug_garbage():
    """
    Turn garbage collection on and disable debug output.
    """
    gc.set_debug(0)
    gc.enable()
