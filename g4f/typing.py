from typing import Any, AsyncGenerator, Generator, NewType, Tuple, TypedDict, Union

SHA256 = NewType('sha_256_hash', str)
CreateResult = Generator[str, None, None]

__all__ = [
    'Any',
    'AsyncGenerator',
    'Generator',
    'Tuple',
    'TypedDict',
    'SHA256',
    'CreateResult',
]