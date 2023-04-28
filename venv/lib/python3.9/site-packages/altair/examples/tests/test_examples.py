import io
import pkgutil

import pytest

from altair.utils.execeval import eval_block
from altair import examples


@pytest.fixture
def require_altair_saver_png():
    try:
        import altair_saver  # noqa: F401
    except ImportError:
        pytest.skip("altair_saver not importable; cannot run saver tests")
    if "png" not in altair_saver.available_formats('vega-lite'):
        pytest.skip("altair_saver not configured to save to png")


def iter_example_filenames():
    for importer, modname, ispkg in pkgutil.iter_modules(examples.__path__):
        if ispkg or modname.startswith('_'):
            continue
        yield modname + '.py'


@pytest.mark.parametrize('filename', iter_example_filenames())
def test_examples(filename: str):
    source = pkgutil.get_data(examples.__name__, filename)
    chart = eval_block(source)

    if chart is None:
        raise ValueError("Example file should define chart in its final "
                         "statement.")
    chart.to_dict()


@pytest.mark.parametrize('filename', iter_example_filenames())
def test_render_examples_to_png(require_altair_saver_png, filename):
    source = pkgutil.get_data(examples.__name__, filename)
    chart = eval_block(source)
    out = io.BytesIO()
    chart.save(out, format="png")
    assert out.getvalue().startswith(b'\x89PNG')
