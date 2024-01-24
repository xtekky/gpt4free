import sys
from typing import Any, AsyncGenerator, Generator, NewType, Tuple, Union, List, Dict, Type, IO, Optional
from PIL.Image import Image

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

SHA256 = NewType('sha_256_hash', str)
CreateResult = Generator[str, None, None]
AsyncResult = AsyncGenerator[str, None]
Messages = List[Dict[str, str]]
ImageType = Union[str, bytes, IO, Image, None]

__all__ = [
    'Any',
    'AsyncGenerator',
    'Generator',
    'Tuple',
    'Union',
    'List',
    'Dict',
    'Type',
    'IO',
    'Optional',
    'TypedDict',
    'SHA256',
    'CreateResult',
    'AsyncResult',
    'Messages',
    'ImageType'
]
