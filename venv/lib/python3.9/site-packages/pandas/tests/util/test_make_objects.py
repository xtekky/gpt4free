"""
Tests for tm.makeFoo functions.
"""


import numpy as np

import pandas._testing as tm


def test_make_multiindex_respects_k():
    # GH#38795 respect 'k' arg
    N = np.random.randint(0, 100)
    mi = tm.makeMultiIndex(k=N)
    assert len(mi) == N
