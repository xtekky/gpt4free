import sys
from typing import Any, AsyncGenerator, Generator, NewType, Tuple, Union

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

SHA256 = NewType('sha_256_hash', str)
CreateResult = Generator[str, None, None]

__all__ = [
    'Any',
    'AsyncGenerator',
    'CreateResult',
    'Generator',
    'SHA256',
    'Tuple',
    'TypedDict',
    'Union',
]
