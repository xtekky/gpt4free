from typing import Any

import numpy as np

from pandas._typing import ExtensionDtype

def check_result_array(obj: object, dtype: np.dtype | ExtensionDtype) -> None: ...
def extract_result(res: object) -> Any: ...
