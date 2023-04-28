"""
Expose a memory-profiling panel to the Django Debug toolbar.

Shows process memory information (virtual size, resident set size) and model
instances for the current request.

Requires Django and Django Debug toolbar:

https://github.com/jazzband/django-debug-toolbar

Pympler adds a memory panel as a third party addon (not included in the
django-debug-toolbar). It can be added by overriding the `DEBUG_TOOLBAR_PANELS`
setting in the Django project settings::

    DEBUG_TOOLBAR_PANELS = (
        'debug_toolbar.panels.timer.TimerDebugPanel',
        'pympler.panels.MemoryPanel',
        )

Pympler also needs to be added to the `INSTALLED_APPS` in the Django settings::

    INSTALLED_APPS = INSTALLED_APPS + ('debug_toolbar', 'pympler')
"""

from pympler.classtracker import ClassTracker
from pympler.process import ProcessMemoryInfo
from pympler.util.stringutils import pp

try:
    from debug_toolbar.panels import Panel
    from django.apps import apps
    from django.template import Context, Template
    from django.template.loader import render_to_string
    from django.http.request import HttpRequest
    from django.http.response import HttpResponse
except ImportError:
    class Panel(object):  # type: ignore
        pass

    class Template(object):  # type: ignore
        pass

    class Context(object):  # type: ignore
        pass

    class HttpRequest(object):  # type: ignore
        pass

    class HttpResponse(object):  # type: ignore
        pass


class MemoryPanel(Panel):

    name = 'pympler'

    title = 'Memory'

    template = 'memory_panel.html'

    classes = [Context, Template]

    def process_request(self, request: HttpRequest) -> HttpResponse:
        self._tracker = ClassTracker()
        for cls in apps.get_models() + self.classes:
            self._tracker.track_class(cls)
        self._tracker.create_snapshot('before')
        self.record_stats({'before': ProcessMemoryInfo()})
        response = super(MemoryPanel, self).process_request(request)
        self.record_stats({'after': ProcessMemoryInfo()})
        self._tracker.create_snapshot('after')
        stats = self._tracker.stats
        stats.annotate()
        self.record_stats({'stats': stats})
        return response

    def enable_instrumentation(self) -> None:
        self._tracker = ClassTracker()
        for cls in apps.get_models() + self.classes:
            self._tracker.track_class(cls)

    def disable_instrumentation(self) -> None:
        self._tracker.detach_all_classes()

    def nav_subtitle(self) -> str:
        context = self.get_stats()
        before = context['before']
        after = context['after']
        rss = after.rss
        delta = rss - before.rss
        delta = ('(+%s)' % pp(delta)) if delta > 0 else ''
        return "%s %s" % (pp(rss), delta)

    @property
    def content(self) -> str:
        context = self.get_stats()
        before = context['before']
        after = context['after']
        stats = context['stats']
        rows = [('Resident set size', after.rss),
                ('Virtual size', after.vsz),
                ]
        rows.extend(after - before)
        rows = [(key, pp(value)) for key, value in rows]
        rows.extend(after.os_specific)

        classes = []
        snapshot = stats.snapshots[-1]
        for model in stats.tracked_classes:
            history = [cnt for _, cnt in stats.history[model]]
            size = snapshot.classes.get(model, {}).get('sum', 0)
            if history and history[-1] > 0:
                classes.append((model, history, pp(size)))
        context.update({'rows': rows, 'classes': classes})
        return render_to_string(self.template, context)
