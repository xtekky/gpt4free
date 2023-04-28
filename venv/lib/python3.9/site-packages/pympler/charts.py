"""
Generate charts from gathered data.

Requires **matplotlib**.
"""

from pympler.classtracker_stats import Stats

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def tracker_timespace(filename: str, stats: Stats) -> None:
        """
        Create a time-space chart from a ``Stats`` instance.
        """
        classlist = list(stats.index.keys())
        classlist.sort()

        for snapshot in stats.snapshots:
            stats.annotate_snapshot(snapshot)

        timestamps = [fp.timestamp for fp in stats.snapshots]
        offsets = [0] * len(stats.snapshots)
        poly_labels = []
        polys = []
        for clsname in classlist:
            pct = [fp.classes[clsname]['pct'] for fp in stats.snapshots
                   if fp.classes and clsname in fp.classes]
            if max(pct) > 3.0:
                sizes = [fp.classes[clsname]['sum'] for fp in stats.snapshots
                         if fp.classes and clsname in fp.classes]
                sizes = [float(x) / (1024 * 1024) for x in sizes]
                sizes = [offset + size for offset, size in zip(offsets, sizes)]
                poly = matplotlib.mlab.poly_between(timestamps, offsets, sizes)
                polys.append((poly, {'label': clsname}))
                poly_labels.append(clsname)
                offsets = sizes

        fig = plt.figure(figsize=(10, 4))
        axis = fig.add_subplot(111)

        axis.set_title("Snapshot Memory")
        axis.set_xlabel("Execution Time [s]")
        axis.set_ylabel("Virtual Memory [MiB]")

        totals = [float(x.asizeof_total) / (1024 * 1024)
                  for x in stats.snapshots]
        axis.plot(timestamps, totals, 'r--', label='Total')
        tracked = [float(x.tracked_total) / (1024 * 1024)
                   for x in stats.snapshots]
        axis.plot(timestamps, tracked, 'b--', label='Tracked total')

        for (args, kwds) in polys:
            axis.fill(*args, **kwds)
        axis.legend(loc=2)  # TODO fill legend
        fig.savefig(filename)

except ImportError:
    def tracker_timespace(filename: str, stats: Stats) -> None:
        pass
