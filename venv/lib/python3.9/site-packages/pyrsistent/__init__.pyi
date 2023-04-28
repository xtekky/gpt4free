# flake8: noqa: E704
# from https://gist.github.com/WuTheFWasThat/091a17d4b5cab597dfd5d4c2d96faf09
# Stubs for pyrsistent (Python 3.6)

from typing import Any
from typing import AnyStr
from typing import Callable
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Mapping
from typing import MutableMapping
from typing import Sequence
from typing import Set
from typing import Union
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import overload

# see commit 08519aa for explanation of the re-export
from pyrsistent.typing import CheckedKeyTypeError as CheckedKeyTypeError
from pyrsistent.typing import CheckedPMap as CheckedPMap
from pyrsistent.typing import CheckedPSet as CheckedPSet
from pyrsistent.typing import CheckedPVector as CheckedPVector
from pyrsistent.typing import CheckedType as CheckedType
from pyrsistent.typing import CheckedValueTypeError as CheckedValueTypeError
from pyrsistent.typing import InvariantException as InvariantException
from pyrsistent.typing import PClass as PClass
from pyrsistent.typing import PBag as PBag
from pyrsistent.typing import PDeque as PDeque
from pyrsistent.typing import PList as PList
from pyrsistent.typing import PMap as PMap
from pyrsistent.typing import PMapEvolver as PMapEvolver
from pyrsistent.typing import PSet as PSet
from pyrsistent.typing import PSetEvolver as PSetEvolver
from pyrsistent.typing import PTypeError as PTypeError
from pyrsistent.typing import PVector as PVector
from pyrsistent.typing import PVectorEvolver as PVectorEvolver

T = TypeVar('T')
KT = TypeVar('KT')
VT = TypeVar('VT')

def pmap(initial: Union[Mapping[KT, VT], Iterable[Tuple[KT, VT]]] = {}, pre_size: int = 0) -> PMap[KT, VT]: ...
def m(**kwargs: VT) -> PMap[str, VT]: ...

def pvector(iterable: Iterable[T] = ...) -> PVector[T]: ...
def v(*iterable: T) -> PVector[T]: ...

def pset(iterable: Iterable[T] = (), pre_size: int = 8) -> PSet[T]: ...
def s(*iterable: T) -> PSet[T]: ...

# see class_test.py for use cases
Invariant = Tuple[bool, Optional[Union[str, Callable[[], str]]]]

@overload
def field(
    type: Union[Type[T], Sequence[Type[T]]] = ...,
    invariant: Callable[[Any], Union[Invariant, Iterable[Invariant]]] = lambda _: (True, None),
    initial: Any = object(),
    mandatory: bool = False,
    factory: Callable[[Any], T] = lambda x: x,
    serializer: Callable[[Any, T], Any] = lambda _, value: value,
) -> T: ...
# The actual return value (_PField) is irrelevant after a PRecord has been instantiated,
# see https://github.com/tobgu/pyrsistent/blob/master/pyrsistent/_precord.py#L10
@overload
def field(
    type: Any = ...,
    invariant: Callable[[Any], Union[Invariant, Iterable[Invariant]]] = lambda _: (True, None),
    initial: Any = object(),
    mandatory: bool = False,
    factory: Callable[[Any], Any] = lambda x: x,
    serializer: Callable[[Any, Any], Any] = lambda _, value: value,
) -> Any: ...

# Use precise types for the simplest use cases, but fall back to Any for
# everything else. See record_test.py for the wide range of possible types for
# item_type
@overload
def pset_field(
    item_type: Type[T],
    optional: bool = False,
    initial: Iterable[T] = ...,
) -> PSet[T]: ...
@overload
def pset_field(
    item_type: Any,
    optional: bool = False,
    initial: Any = (),
) -> PSet[Any]: ...

@overload
def pmap_field(
    key_type: Type[KT],
    value_type: Type[VT],
    optional: bool = False,
    invariant: Callable[[Any], Tuple[bool, Optional[str]]] = lambda _: (True, None),
) -> PMap[KT, VT]: ...
@overload
def pmap_field(
    key_type: Any,
    value_type: Any,
    optional: bool = False,
    invariant: Callable[[Any], Tuple[bool, Optional[str]]] = lambda _: (True, None),
) -> PMap[Any, Any]: ...

@overload
def pvector_field(
    item_type: Type[T],
    optional: bool = False,
    initial: Iterable[T] = ...,
) -> PVector[T]: ...
@overload
def pvector_field(
    item_type: Any,
    optional: bool = False,
    initial: Any = (),
) -> PVector[Any]: ...

def pbag(elements: Iterable[T]) -> PBag[T]: ...
def b(*elements: T) -> PBag[T]: ...

def plist(iterable: Iterable[T] = (), reverse: bool = False) -> PList[T]: ...
def l(*elements: T) -> PList[T]: ...

def pdeque(iterable: Optional[Iterable[T]] = None, maxlen: Optional[int] = None) -> PDeque[T]: ...
def dq(*iterable: T) -> PDeque[T]: ...

@overload
def optional(type: T) -> Tuple[T, Type[None]]: ...
@overload
def optional(*typs: Any) -> Tuple[Any, ...]: ...

T_PRecord = TypeVar('T_PRecord', bound='PRecord')
class PRecord(PMap[AnyStr, Any]):
    _precord_fields: Mapping
    _precord_initial_values: Mapping

    def __hash__(self) -> int: ...
    def __init__(self, **kwargs: Any) -> None: ...
    def __iter__(self) -> Iterator[Any]: ...
    def __len__(self) -> int: ...
    @classmethod
    def create(
        cls: Type[T_PRecord],
        kwargs: Mapping,
        _factory_fields: Optional[Iterable] = None,
        ignore_extra: bool = False,
    ) -> T_PRecord: ...
    # This is OK because T_PRecord is a concrete type
    def discard(self: T_PRecord, key: KT) -> T_PRecord: ...
    def remove(self: T_PRecord, key: KT) -> T_PRecord: ...

    def serialize(self, format: Optional[Any] = ...) -> MutableMapping: ...

    # From pyrsistent documentation:
    #   This set function differs slightly from that in the PMap
    #   class. First of all it accepts key-value pairs. Second it accepts multiple key-value
    #   pairs to perform one, atomic, update of multiple fields.
    @overload
    def set(self, key: KT, val: VT) -> Any: ...
    @overload
    def set(self, **kwargs: VT) -> Any: ...

def immutable(
    members: Union[str, Iterable[str]] = '',
    name: str = 'Immutable',
    verbose: bool = False,
) -> Tuple: ...  # actually a namedtuple

# ignore mypy warning "Overloaded function signatures 1 and 5 overlap with
# incompatible return types"
@overload
def freeze(o: Mapping[KT, VT]) -> PMap[KT, VT]: ... # type: ignore
@overload
def freeze(o: List[T]) -> PVector[T]: ... # type: ignore
@overload
def freeze(o: Tuple[T, ...]) -> Tuple[T, ...]: ...
@overload
def freeze(o: Set[T]) -> PSet[T]: ... # type: ignore
@overload
def freeze(o: T) -> T: ...


@overload
def thaw(o: PMap[KT, VT]) -> MutableMapping[KT, VT]: ... # type: ignore
@overload
def thaw(o: PVector[T]) -> List[T]: ... # type: ignore
@overload
def thaw(o: Tuple[T, ...]) -> Tuple[T, ...]: ...
# collections.abc.MutableSet is kind of garbage:
# https://stackoverflow.com/questions/24977898/why-does-collections-mutableset-not-bestow-an-update-method
@overload
def thaw(o: PSet[T]) -> Set[T]: ... # type: ignore
@overload
def thaw(o: T) -> T: ...

def mutant(fn: Callable) -> Callable: ...

def inc(x: int) -> int: ...
@overload
def discard(evolver: PMapEvolver[KT, VT], key: KT) -> None: ...
@overload
def discard(evolver: PVectorEvolver[T], key: int) -> None: ...
@overload
def discard(evolver: PSetEvolver[T], key: T) -> None: ...
def rex(expr: str) -> Callable[[Any], bool]: ...
def ny(_: Any) -> bool: ...

def get_in(keys: Iterable, coll: Mapping, default: Optional[Any] = None, no_default: bool = False) -> Any: ...
