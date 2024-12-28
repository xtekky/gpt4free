import sys
import os
from typing import Any, AsyncGenerator, Generator, AsyncIterator, Iterator, NewType, Tuple, Union, List, Dict, Type, IO, Optional

try:
    from PIL.Image import Image
except ImportError:
    class Image:
        pass

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict
    
from .providers.response import ResponseType

SHA256 = NewType('sha_256_hash', str)
CreateResult = Iterator[Union[str, ResponseType]]
AsyncResult = AsyncIterator[Union[str, ResponseType]]
Messages = List[Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]]]
Cookies = Dict[str, str]
ImageType = Union[str, bytes, IO, Image, os.PathLike]
ImagesType = List[Tuple[ImageType, Optional[str]]]

__all__ = [
    'Any',
    'AsyncGenerator',
    'Generator',
    'AsyncIterator',
    'Iterator'
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
    'Cookies',
    'Image',
    'ImageType',
    'ImagesType'
]
