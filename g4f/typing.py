from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    List,
    NewType,
    Tuple,
    TypedDict,
    Union,
)

SHA256 = NewType('sha_256_hash', str)
CreateResult = Generator[str, None, None]

__all__ = [
    'Any',
    'AsyncGenerator',
    'Dict',
    'Generator',
    'List',
    'Tuple',
    'TypedDict',
    'SHA256',
    'CreateResult',
]