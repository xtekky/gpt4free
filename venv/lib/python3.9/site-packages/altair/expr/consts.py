from .core import ConstExpression


CONST_LISTING = {
    "NaN": "not a number (same as JavaScript literal NaN)",
    "LN10": "the natural log of 10 (alias to Math.LN10)",
    "E": "the transcendental number e (alias to Math.E)",
    "LOG10E": "the base 10 logarithm e (alias to Math.LOG10E)",
    "LOG2E": "the base 2 logarithm of e (alias to Math.LOG2E)",
    "SQRT1_2": "the square root of 0.5 (alias to Math.SQRT1_2)",
    "LN2": "the natural log of 2 (alias to Math.LN2)",
    "SQRT2": "the square root of 2 (alias to Math.SQRT1_2)",
    "PI": "the transcendental number pi (alias to Math.PI)",
}

NAME_MAP = {}


def _populate_namespace():
    globals_ = globals()
    for name, doc in CONST_LISTING.items():
        py_name = NAME_MAP.get(name, name)
        globals_[py_name] = ConstExpression(name, doc)
        yield py_name


__all__ = list(_populate_namespace())
