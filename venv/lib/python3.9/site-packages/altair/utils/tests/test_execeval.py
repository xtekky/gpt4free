from ..execeval import eval_block

HAS_RETURN = """
x = 4
y = 2 * x
3 * y
"""

NO_RETURN = """
x = 4
y = 2 * x
z = 3 * y
"""


def test_eval_block_with_return():
    _globals = {}
    result = eval_block(HAS_RETURN, _globals)
    assert result == 24
    assert _globals["x"] == 4
    assert _globals["y"] == 8


def test_eval_block_without_return():
    _globals = {}
    result = eval_block(NO_RETURN, _globals)
    assert result is None
    assert _globals["x"] == 4
    assert _globals["y"] == 8
    assert _globals["z"] == 24
