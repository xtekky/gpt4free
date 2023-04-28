#!/usr/bin/env python

# Copyright, license and disclaimer are at the very end of this file.

# This is the latest, enhanced version of the asizeof.py recipes at
# <http://ASPN.ActiveState.com/ASPN/Cookbook/Python/Recipe/546530>
# <http://ASPN.ActiveState.com/ASPN/Cookbook/Python/Recipe/544288>

# Note, objects like ``namedtuples``, ``closure``, and NumPy data
# ``arange``, ``array``, ``matrix``, etc. are only handled by recent
# versions of this module.  Sizing of ``__slots__`` has been incorrect
# in versions before this one.  Also, property ``Asizer.duplicate`` gave
# incorrect values before this release.  Several other properties
# have been added to the ``Asizer`` class and the ``print_summary``
# method has been updated.

'''
This module exposes 9 functions and 2 classes to obtain lengths and
sizes of Python objects (for Python 3.5 or later).

Earlier versions of this module supported Python versions down to
Python 2.2.  If you are using Python 3.5 or older, please consider
downgrading Pympler.

**Public Functions** [#unsafe]_

   Function **asizeof** calculates the combined (approximate) size
   in bytes of one or several Python objects.

   Function **asizesof** returns a tuple containing the (approximate)
   size in bytes for each given Python object separately.

   Function **asized** returns for each object an instance of class
   **Asized** containing all the size information of the object and
   a tuple with the referents [#refs]_.

   Functions **basicsize** and **itemsize** return the *basic-*
   respectively *itemsize* of the given object, both in bytes.  For
   objects as ``array.array``, ``numpy.array``, ``numpy.matrix``,
   etc. where the item size varies depending on the instance-specific
   data type, function **itemsize** returns that item size.

   Function **flatsize** returns the *flat size* of a Python object
   in bytes defined as the *basic size* plus the *item size* times
   the *length* of the given object.

   Function **leng** returns the *length* of an object, like standard
   function ``len`` but extended for several types.  E.g. the **leng**
   of a multi-precision int (or long) is the number of ``digits``
   [#digit]_.  The length of most *mutable* sequence objects includes
   an estimate of the over-allocation and therefore, the **leng** value
   may differ from the standard ``len`` result.  For objects like
   ``array.array``, ``numpy.array``, ``numpy.matrix``, etc. function
   **leng** returns the proper number of items.

   Function **refs** returns (a generator for) the referents [#refs]_
   of the given object.

   Certain classes are known to be sub-classes of or to behave as
   ``dict`` objects.  Function **adict** can be used to register
   other class objects to be treated like ``dict``.

**Public Classes** [#unsafe]_

   Class **Asizer** may be used to accumulate the results of several
   **asizeof** or **asizesof** calls.  After creating an **Asizer**
   instance, use methods **asizeof** and **asizesof** as needed to
   size any number of additional objects.

   Call methods **exclude_refs** and/or **exclude_types** to exclude
   references to respectively instances or types of certain objects.

   Use one of the **print\\_...** methods to report the statistics.

   An instance of class **Asized** is returned for each object sized
   by the **asized** function or method.

**Duplicate Objects**

   Any duplicate, given objects are sized only once and the size
   is included in the accumulated total only once.  But functions
   **asizesof** and **asized** will return a size value respectively
   an **Asized** instance for each given object, including duplicates.

**Definitions** [#arb]_

   The *length* of an objects like ``dict``, ``list``, ``set``,
   ``str``, ``tuple``, etc. is defined as the number of items held
   in or allocated by the object.  Held items are *references* to
   other objects, called the *referents*.

   The *size* of an object is defined as the sum of the *flat size*
   of the object plus the sizes of any referents [#refs]_.  Referents
   are visited recursively up to the specified detail level.  However,
   the size of objects referenced multiple times is included only once
   in the total *size*.

   The *flat size* of an object is defined as the *basic size* of the
   object plus the *item size* times the number of allocated *items*,
   *references* to referents.  The *flat size* does include the size
   for the *references* to the referents, but not the size of the
   referents themselves.

   The *flat size* returned by function *flatsize* equals the result
   of function *asizeof* with options *code=True*, *ignored=False*,
   *limit=0* and option *align* set to the same value.

   The accurate *flat size* for an object is obtained from function
   ``sys.getsizeof()`` where available.  Otherwise, the *length* and
   *size* of sequence objects as ``dicts``, ``lists``, ``sets``, etc.
   is based on an estimate for the number of allocated items.  As a
   result, the reported *length* and *size* may differ substantially
   from the actual *length* and *size*.

   The *basic* and *item size* are obtained from the ``__basicsize__``
   respectively ``__itemsize__`` attributes of the (type of the)
   object.  Where necessary (e.g. sequence objects), a zero
   ``__itemsize__`` is replaced by the size of a corresponding C type.

   The overhead for Python's garbage collector (GC) is included in
   the *basic size* of (GC managed) objects as well as the space
   needed for ``refcounts`` (used only in certain Python builds).

   Optionally, size values can be aligned to any power-of-2 multiple.

**Size of (byte)code**

   The *(byte)code size* of objects like classes, functions, methods,
   modules, etc. can be included by setting option *code=True*.

   Iterators are handled like sequences: iterated object(s) are sized
   like *referents* [#refs]_, but only up to the specified level or
   recursion *limit* (and only if function ``gc.get_referents()``
   returns the referent object of iterators).

   Generators are sized as *(byte)code* only, but the generated
   objects are never sized.

**Old- and New-style Classes**

   All old- and new-style ``class``, instance and ``type`` objects are
   handled uniformly such that (a) instance objects are distinguished
   from class objects and (b) instances of different old-style classes
   can be dealt with separately.

   Class and type objects are represented as ``<class ....* def>``
   respectively ``<type ... def>`` where the ``*`` indicates an old-style
   class and the ``... def`` suffix marks the *definition object*.
   Instances of  classes are shown as ``<class module.name*>`` without
   the ``... def`` suffix.  The ``*`` after the name indicates an
   instance of an old-style class.

**Ignored Objects**

   To avoid excessive sizes, several object types are ignored [#arb]_
   by default, e.g. built-in functions, built-in types and classes
   [#bi]_, function globals and module referents.  However, any
   instances thereof and module objects will be sized when passed as
   given objects.  Ignored object types are included unless option
   *ignored* is set accordingly.

   In addition, many ``__...__`` attributes of callable objects are
   ignored [#arb]_, except crucial ones, e.g. class attributes ``__dict__``,
   ``__doc__``, ``__name__`` and ``__slots__``.  For more details, see
   the type-specific ``_..._refs()`` and ``_len_...()`` functions below.

.. rubric:: Footnotes
.. [#unsafe] The functions and classes in this module are not thread-safe.

.. [#refs] The *referents* of an object are the objects referenced *by*
     that object.  For example, the *referents* of a ``list`` are the
     objects held in the ``list``, the *referents* of a ``dict`` are
     the key and value objects in the ``dict``, etc.

.. [#arb] These definitions and other assumptions are rather arbitrary
     and may need corrections or adjustments.

.. [#digit] See Python source file ``.../Include/longinterp.h`` for the
     C ``typedef`` of ``digit`` used in multi-precision int (or long)
     objects.  The C ``sizeof(digit)`` in bytes can be obtained in
     Python from the int (or long) ``__itemsize__`` attribute.
     Function **leng** determines the number of ``digits`` of an int
     (or long) object.

.. [#bi] ``Type``s and ``class``es are considered built-in if the
     ``__module__`` of the type or class is listed in the private
     ``_builtin_modules``.
'''  # PYCHOK escape
import sys
if sys.version_info < (3, 6, 0):
    raise NotImplementedError('%s requires Python 3.6 or newer' % ('asizeof',))

from typing import Callable, Dict, List, Optional, Tuple, Union

# all imports listed explicitly to help PyChecker
from inspect import (isbuiltin, isclass, iscode, isframe, isfunction,
                     ismethod, ismodule, stack)
from math import log
from os import curdir, linesep
from struct import calcsize  # type/class Struct only in Python 2.5+
import types as Types
import warnings
import weakref as Weakref

__all__ = ['adict', 'asized', 'asizeof', 'asizesof',
           'Asized', 'Asizer',  # classes
           'basicsize', 'flatsize', 'itemsize', 'leng', 'refs']
__version__ = '21.08.09'

# Any classes or types in modules listed in _builtin_modules are
# considered built-in and ignored by default, as built-in functions
_builtin_mods = [int.__module__, 'types', Exception.__module__]  # 'weakref'
if __name__ != '__main__':  # treat this very module as built-in
    _builtin_mods.append(__name__)
_builtin_modules = tuple(_builtin_mods)

# Sizes of some primitive C types
# XXX len(pack(T, 0)) == Struct(T).size == calcsize(T)
_sizeof_Cbyte  = calcsize('c')  # sizeof(unsigned char)
_sizeof_Clong  = calcsize('l')  # sizeof(long)
_sizeof_Cvoidp = calcsize('P')  # sizeof(void*)

# sizeof(long) != sizeof(ssize_t) on LLP64
if _sizeof_Clong < _sizeof_Cvoidp:  # pragma: no coverage
    _z_P_L = 'P'
else:
    _z_P_L = 'L'


def _calcsize(fmt):
    '''Like struct.calcsize() but handling 'z' for Py_ssize_t.
    '''
    return calcsize(fmt.replace('z', _z_P_L))


# Defaults for some basic sizes with 'z' for C Py_ssize_t
_sizeof_CPyCodeObject = _calcsize('Pz10P5i0P')  # sizeof(PyCodeObject)
_sizeof_CPyFrameObject = _calcsize('Pzz13P63i0P')  # sizeof(PyFrameObject)
_sizeof_CPyModuleObject = _calcsize('PzP0P')  # sizeof(PyModuleObject)

# Defaults for some item sizes with 'z' for C Py_ssize_t
_sizeof_CPyDictEntry = _calcsize('z2P')  # sizeof(PyDictEntry)
_sizeof_Csetentry = _calcsize('lP')  # sizeof(setentry)

_sizeof_Cdigit = int.__itemsize__
if _sizeof_Cdigit < 2:  # pragma: no coverage
    raise AssertionError('sizeof(%s) bad: %d' % ('digit', _sizeof_Cdigit))

_builtins2 = (range,)

# Get character size for internal unicode representation in Python < 3.3
u = '\0'.encode('utf-8')
_sizeof_Cunicode = len(u)
del u

try:  # Size of GC header, sizeof(PyGC_Head)
    import _testcapi as t
    _sizeof_CPyGC_Head = t.SIZEOF_PYGC_HEAD  # new in Python 2.6
except (ImportError, AttributeError):  # sizeof(PyGC_Head)
    # alignment should be to sizeof(long double) but there
    # is no way to obtain that value, assume twice double
    t = calcsize('2d') - 1
    _sizeof_CPyGC_Head = (_calcsize('2Pz') + t) & ~t
del t

# Size of refcounts (Python debug build only)
if hasattr(sys, 'gettotalrefcount'):  # pragma: no coverage
    _sizeof_Crefcounts = _calcsize('2z')
else:
    _sizeof_Crefcounts = 0

from abc import ABCMeta

# Some flags from .../Include/object.h
_Py_TPFLAGS_HEAPTYPE = 1 << 9  # Py_TPFLAGS_HEAPTYPE
_Py_TPFLAGS_HAVE_GC = 1 << 14  # Py_TPFLAGS_HAVE_GC

_Type_type = type(type)  # == type and new-style class type


# Compatibility functions for more uniform
# behavior across Python version 2.2 thu 3+

def _items(obj):  # dict only
    '''Return iter-/generator, preferably.
    '''
    o = getattr(obj, 'iteritems', obj.items)
    if _callable(o):
        return o()
    else:
        return o or ()


def _keys(obj):  # dict only
    '''Return iter-/generator, preferably.
    '''
    o = getattr(obj, 'iterkeys', obj.keys)
    if _callable(o):
        return o()
    else:
        return o or ()


def _values(obj):  # dict only
    '''Return iter-/generator, preferably.
    '''
    o = getattr(obj, 'itervalues', obj.values)
    if _callable(o):
        return o()
    else:
        return o or ()


try:  # callable() builtin
    _callable = callable
except NameError:  # callable() removed in Python 3+
    def _callable(obj):
        '''Substitute for callable().'''
        return hasattr(obj, '__call__')

# 'cell' is holding data used in closures
c = (lambda unused: (lambda: unused))(None)
_cell_type = type(c.__closure__[0])  # type: ignore
del c

from gc import get_objects as _getobjects  # containers only?

if sys.platform == 'ios':  # Apple iOS
    _gc_getobjects = _getobjects

    def _getobjects():  # PYCHOK expected
        # avoid Pythonista3/Python 3+ crash
        return tuple(o for o in _gc_getobjects() if not _isNULL(o))

from gc import get_referents as _getreferents

# sys.getsizeof() new in Python 2.6
_getsizeof = sys.getsizeof  # overridden below
_getsizeof_excls = ()  # types not sys.getsizeof'd

from sys import intern as _intern


# Private functions

def _basicsize(t, base=0, heap=False, obj=None):
    '''Get non-zero basicsize of type,
       including the header sizes.
    '''
    s = max(getattr(t, '__basicsize__', 0), base)
    # include gc header size
    if t != _Type_type:
        h = getattr(t, '__flags__', 0) & _Py_TPFLAGS_HAVE_GC
    elif heap:  # type, allocated on heap
        h = True
    else:  # None has no __flags__ attr
        h = getattr(obj, '__flags__', 0) & _Py_TPFLAGS_HEAPTYPE
    if h:
        s += _sizeof_CPyGC_Head
    # include reference counters
    return s + _sizeof_Crefcounts


def _classof(obj, dflt=None):
    '''Return the object's class object.
    '''
    return getattr(obj, '__class__', dflt)


def _derive_typedef(typ):
    '''Return single, existing super type typedef or None.
    '''
    v = [v for v in _values(_typedefs) if _issubclass(typ, v.type)]
    if len(v) == 1:
        return v[0]
    return None


def _dir2(obj, pref='', excl=(), slots=None, itor=''):
    '''Return an attribute name, object 2-tuple for certain
       attributes or for the ``__slots__`` attributes of the
       given object, but not both.  Any iterator referent
       objects are returned with the given name if the
       latter is non-empty.
    '''
    if slots:  # __slots__ attrs
        if hasattr(obj, slots):
            # collect all inherited __slots__ attrs
            # from list, tuple, or dict __slots__,
            # while removing any duplicate attrs
            s = {}
            for c in type(obj).mro():
                for a in getattr(c, slots, ()):
                    if a.startswith('__'):
                        a = '_' + c.__name__ + a
                    if hasattr(obj, a):
                        s.setdefault(a, getattr(obj, a))
            # assume __slots__ tuple-like is holding the values
            # yield slots, _Slots(s)  # _keys(s) ... REMOVED,
            # see _Slots.__doc__ further below
            for t in _items(s):
                yield t  # attr name, value
    elif itor:  # iterator referents
        for o in obj:  # iter(obj)
            yield itor, o
    else:  # regular attrs
        for a in dir(obj):
            if a.startswith(pref) and hasattr(obj, a) and a not in excl:
                yield a, getattr(obj, a)


def _getsizeof_excls_add(typ):
    '''Add another type to the tuple of types to be
       excluded from sys.getsizeof due to errors.
    '''
    global _getsizeof_excls
    if typ and typ not in _getsizeof_excls:
        _getsizeof_excls += (typ,)


def _infer_dict(obj):
    '''Return True for likely dict object via duck typing.
    '''
    for attrs in (('items', 'keys', 'values'),  # 'update',
                  ('iteritems', 'iterkeys', 'itervalues')):
        attrs += '__len__', 'get', 'has_key'
        if all(_callable(getattr(obj, a, None)) for a in attrs):
            return True
    return False


def _isbuiltin2(obj):
    '''Return True for builtins like Python 2.
    '''
    # range is no longer a built-in in Python 3+
    return isbuiltin(obj) or obj in _builtins2


def _iscell(obj):
    '''Return True if obj is a cell as used in a closure.
    '''
    return isinstance(obj, _cell_type)


def _isdictclass(obj):
    '''Return True for known dict objects.
    '''
    c = _classof(obj)
    return c and c.__name__ in _dict_classes.get(c.__module__, ())


def _isframe(obj):
    '''Return True for a stack frame object.
    '''
    try:  # safe isframe(), see pympler.muppy
        return isframe(obj)
    except ReferenceError:
        return False


def _isnamedtuple(obj):
    '''Named tuples are identified via duck typing:
       <http://www.gossamer-threads.com/lists/python/dev/1142178>
    '''
    return isinstance(obj, tuple) and hasattr(obj, '_fields')


def _isNULL(obj):
    '''Prevent asizeof(all=True, ...) crash.

       Sizing gc.get_objects() crashes in Pythonista3 with
       Python 3.5.1 on iOS due to 1-tuple (<Null>,) object,
       see <http://forum.omz-software.com/user/mrjean1>.
    '''
    return isinstance(obj, tuple) and len(obj) == 1 \
                                  and repr(obj) == '(<NULL>,)'


def _issubclass(sub, sup):
    '''Safe issubclass().
    '''
    if sup is not object:
        try:
            return issubclass(sub, sup)
        except TypeError:
            pass
    return False


def _itemsize(t, item=0):
    '''Get non-zero itemsize of type.
    '''
    # replace zero value with default
    return getattr(t, '__itemsize__', 0) or item


def _kwdstr(**kwds):
    '''Keyword arguments as a string.
    '''
    return ', '.join(sorted('%s=%r' % kv for kv in _items(kwds)))


def _lengstr(obj):
    '''Object length as a string.
    '''
    n = leng(obj)
    if n is None:  # no len
        r = ''
    elif n > _len(obj):  # extended
        r = ' leng %d!' % n
    else:
        r = ' leng %d' % n
    return r


def _moduleof(obj, dflt=''):
    '''Return the object's module name.
    '''
    return getattr(obj, '__module__', dflt)


def _nameof(obj, dflt=''):
    '''Return the name of an object.
    '''
    return getattr(obj, '__name__', dflt)


def _objs_opts_x(objs, all=None, **opts):
    '''Return given or 'all' objects
       and the remaining options.
    '''
    if objs:  # given objects
        t = objs
        x = False
    elif all in (False, None):
        t = ()
        x = True
    elif all is True:  # 'all' objects
        t = _getobjects()
        x = True
    else:
        raise ValueError('invalid option: %s=%r' % ('all', all))
    return t, opts, x


def _p100(part, total, prec=1):
    '''Return percentage as string.
    '''
    r = float(total)
    if r:
        r = part * 100.0 / r
        return '%.*f%%' % (prec, r)
    return 'n/a'


def _plural(num):
    '''Return 's' if plural.
    '''
    if num == 1:
        s = ''
    else:
        s = 's'
    return s


def _power2(n):
    '''Find the next power of 2.
    '''
    p2 = 16
    while n > p2:
        p2 += p2
    return p2


def _prepr(obj, clip=0):
    '''Prettify and clip long repr() string.
    '''
    return _repr(obj, clip=clip).strip('<>').replace("'", '')  # remove <''>


def _printf(fmt, *args, **print3options):
    '''Formatted print to sys.stdout or given stream.

       *print3options* -- some keyword arguments, like Python 3+ print.
    '''
    if print3options:  # like Python 3+
        f = print3options.get('file', None) or sys.stdout
        if args:
            f.write(fmt % args)
        else:
            f.write(fmt)
        f.write(print3options.get('end', linesep))
        if print3options.get('flush', False):
            f.flush()
    elif args:
        print(fmt % args)
    else:
        print(fmt)


def _refs(obj, named, *attrs, **kwds):
    '''Return specific attribute objects of an object.
    '''
    if named:
        _N = _NamedRef
    else:
        def _N(_, o):
            return o

    for a in attrs:  # cf. inspect.getmembers()
        if hasattr(obj, a):
            yield _N(a, getattr(obj, a))
    if kwds:  # kwds are _dir2() args
        for a, o in _dir2(obj, **kwds):
            yield _N(a, o)


def _repr(obj, clip=80):
    '''Clip long repr() string.
    '''
    try:  # safe repr()
        r = repr(obj).replace(linesep, '\\n')
    except Exception:
        r = 'N/A'
    if len(r) > clip > 0:
        h = (clip // 2) - 2
        if h > 0:
            r = r[:h] + '....' + r[-h:]
    return r


def _SI(size, K=1024, i='i'):
    '''Return size as SI string.
    '''
    if 1 < K <= size:
        f = float(size)
        for si in iter('KMGPTE'):
            f /= K
            if f < K:
                return ' or %.1f %s%sB' % (f, si, i)
    return ''


def _SI2(size, **kwds):
    '''Return size as regular plus SI string.
    '''
    return str(size) + _SI(size, **kwds)


# Type-specific referents functions

def _cell_refs(obj, named):
    try:  # handle 'empty' cells
        o = obj.cell_contents
        if named:
            o = _NamedRef('cell_contents', o)
        yield o
    except (AttributeError, ValueError):
        pass


def _class_refs(obj, named):
    '''Return specific referents of a class object.
    '''
    return _refs(obj, named, '__class__', '__doc__', '__mro__',
                             '__name__', '__slots__', '__weakref__',
                             '__dict__')  # __dict__ last


def _co_refs(obj, named):
    '''Return specific referents of a code object.
    '''
    return _refs(obj, named, pref='co_')


def _dict_refs(obj, named):
    '''Return key and value objects of a dict/proxy.
    '''
    try:
        if named:
            for k, v in _items(obj):
                s = str(k)
                yield _NamedRef('[K] ' + s, k)
                yield _NamedRef('[V] ' + s + ': ' + _repr(v), v)
        else:
            for k, v in _items(obj):
                yield k
                yield v
    except (KeyError, ReferenceError, TypeError) as x:
        warnings.warn("Iterating '%s': %r" % (_classof(obj), x))


def _enum_refs(obj, named):
    '''Return specific referents of an enumerate object.
    '''
    return _refs(obj, named, '__doc__')


def _exc_refs(obj, named):
    '''Return specific referents of an Exception object.
    '''
    # .message raises DeprecationWarning in Python 2.6
    return _refs(obj, named, 'args', 'filename', 'lineno', 'msg', 'text')  # , 'message', 'mixed'


def _file_refs(obj, named):
    '''Return specific referents of a file object.
    '''
    return _refs(obj, named, 'mode', 'name')


def _frame_refs(obj, named):
    '''Return specific referents of a frame object.
    '''
    return _refs(obj, named, pref='f_')


def _func_refs(obj, named):
    '''Return specific referents of a function or lambda object.
    '''
    return _refs(obj, named, '__doc__', '__name__', '__code__', '__closure__',
                 pref='func_', excl=('func_globals',))


def _gen_refs(obj, named):
    '''Return the referent(s) of a generator (expression) object.
    '''
    # only some gi_frame attrs
    f = getattr(obj, 'gi_frame', None)
    return _refs(f, named, 'f_locals', 'f_code')
#   do not yield any items to keep generator intact
#   for r in _refs(f, named, 'f_locals', 'f_code'):
#       yield r
#   for r in obj:
#       yield r


def _im_refs(obj, named):
    '''Return specific referents of a method object.
    '''
    return _refs(obj, named, '__doc__', '__name__', '__code__', pref='im_')


def _inst_refs(obj, named):
    '''Return specific referents of a class instance.
    '''
    return _refs(obj, named, '__dict__', '__class__', slots='__slots__')


def _iter_refs(obj, named):
    '''Return the referent(s) of an iterator object.
    '''
    r = _getreferents(obj)  # special case
    return _refs(r, named, itor=_nameof(obj) or 'iteref')


def _module_refs(obj, named):
    '''Return specific referents of a module object.
    '''
    # ignore this very module
    if obj.__name__ == __name__:
        return ()
    # module is essentially a dict
    return _dict_refs(obj.__dict__, named)


def _namedtuple_refs(obj, named):
    '''Return specific referents of obj-as-sequence and slots but exclude dict.
    '''
    for r in _refs(obj, named, '__class__', slots='__slots__'):
        yield r
    for r in obj:
        yield r


def _prop_refs(obj, named):
    '''Return specific referents of a property object.
    '''
    return _refs(obj, named, '__doc__', pref='f')


def _seq_refs(obj, unused):  # named unused for PyChecker
    '''Return specific referents of a frozen/set, list, tuple and xrange object.
    '''
    return obj  # XXX for r in obj: yield r


def _stat_refs(obj, named):
    '''Return referents of a os.stat object.
    '''
    return _refs(obj, named, pref='st_')


def _statvfs_refs(obj, named):
    '''Return referents of a os.statvfs object.
    '''
    return _refs(obj, named, pref='f_')


def _tb_refs(obj, named):
    '''Return specific referents of a traceback object.
    '''
    return _refs(obj, named, pref='tb_')


def _type_refs(obj, named):
    '''Return specific referents of a type object.
    '''
    return _refs(obj, named, '__doc__', '__mro__', '__name__',
                             '__slots__', '__weakref__', '__dict__')


def _weak_refs(obj, unused):  # named unused for PyChecker
    '''Return weakly referent object.
    '''
    try:  # ignore 'key' of KeyedRef
        return (obj(),)
    except Exception:  # XXX ReferenceError
        return ()


_all_refs = (None, _cell_refs, _class_refs, _co_refs, _dict_refs, _enum_refs,
                   _exc_refs, _file_refs, _frame_refs, _func_refs, _gen_refs,
                   _im_refs, _inst_refs, _iter_refs, _module_refs, _namedtuple_refs,
                   _prop_refs, _seq_refs, _stat_refs, _statvfs_refs, _tb_refs,
                   _type_refs, _weak_refs)


# Type-specific length functions

def _len(obj):
    '''Safe len().
    '''
    try:
        return len(obj)
    except TypeError:  # no len()
        return 0


def _len_bytearray(obj):
    '''Bytearray size.
    '''
    return obj.__alloc__()


def _len_code(obj):  # see .../Lib/test/test_sys.py
    '''Length of code object (stack and variables only).
    '''
    return (obj.co_stacksize + obj.co_nlocals +
           _len(obj.co_freevars) + _len(obj.co_cellvars) - 1)


def _len_dict(obj):
    '''Dict length in items (estimate).
    '''
    n = len(obj)  # active items
    if n < 6:  # ma_smalltable ...
        n = 0  # ... in basicsize
    else:  # at least one unused
        n = _power2(n + 1)
    return n


def _len_frame(obj):
    '''Length of a frame object.
    '''
    c = getattr(obj, 'f_code', None)
    if c:
        n = _len_code(c)
    else:
        n = 0
    return n


_digit2p2 = 1 << (_sizeof_Cdigit << 3)
_digitmax = _digit2p2 - 1  # == (2 * PyLong_MASK + 1)
_digitlog = 1.0 / log(_digit2p2)


def _len_int(obj):
    '''Length of multi-precision int (aka long) in digits.
    '''
    if obj:
        n, i = 1, abs(obj)
        if i > _digitmax:
            # no log(x[, base]) in Python 2.2
            n += int(log(i) * _digitlog)
    else:  # zero
        n = 0
    return n


def _len_iter(obj):
    '''Length (hint) of an iterator.
    '''
    n = getattr(obj, '__length_hint__', None)
    if n:
        n = n()
    else:  # try len()
        n = _len(obj)
    return n


def _len_list(obj):
    '''Length of list (estimate).
    '''
    n = len(obj)
    # estimate over-allocation
    if n > 8:
        n += 6 + (n >> 3)
    elif n:
        n += 4
    return n


def _len_module(obj):
    '''Module length.
    '''
    return _len(obj.__dict__)  # _len(dir(obj))


def _len_set(obj):
    '''Length of frozen/set (estimate).
    '''
    n = len(obj)
    if n > 8:  # assume half filled
        n = _power2(n + n - 2)
    elif n:  # at least 8
        n = 8
    return n


def _len_slice(obj):
    '''Slice length.
    '''
    try:
        return ((obj.stop - obj.start + 1) // obj.step)
    except (AttributeError, TypeError):
        return 0


# REMOVED, see _Slots.__doc__
# def _len_slots(obj):
#     '''Slots length.
#     '''
#     return len(obj) - 1


def _len_struct(obj):
    '''Struct length in bytes.
    '''
    try:
        return obj.size
    except AttributeError:
        return 0


def _len_unicode(obj):
    '''Unicode size.
    '''
    return len(obj) + 1


_all_lens = (None, _len, _len_bytearray, _len_code, _len_dict,
                   _len_frame, _len_int, _len_iter, _len_list,
                   _len_module, _len_set, _len_slice, _len_struct,
                   _len_unicode)  # type: Tuple[Union[None, Callable], ...] # _len_array, _len_numpy, _len_slots


# More private functions and classes

_old_style = '*'  # marker
_new_style = ''   # no marker


class _Claskey(object):
    '''Wrapper for class objects.
    '''
    __slots__ = ('_obj', '_sty')

    def __init__(self, obj, style):
        self._obj = obj  # XXX Weakref.ref(obj)
        self._sty = style

    def __str__(self):
        r = str(self._obj)
        if r.endswith('>'):
            r = '%s%s def>' % (r[:-1], self._sty)
        elif self._sty is _old_style and not r.startswith('class '):
            r = 'class %s%s def' % (r, self._sty)
        else:
            r = '%s%s def' % (r, self._sty)
        return r
    __repr__ = __str__


# For most objects, the object type is used as the key in the
# _typedefs dict further below, except class and type objects
# and old-style instances.  Those are wrapped with separate
# _Claskey or _Instkey instances to be able (1) to distinguish
# instances of different old-style classes by class, (2) to
# distinguish class (and type) instances from class (and type)
# definitions for new-style classes and (3) provide similar
# results for repr() and str() of new- and old-style classes
# and instances.

_claskeys = {}  # type: Dict[int, _Claskey]


def _claskey(obj, style):
    '''Wrap an old- or new-style class object.
    '''
    i = id(obj)
    k = _claskeys.get(i, None)
    if not k:
        _claskeys[i] = k = _Claskey(obj, style)
    return k


def _keytuple(obj):  # PYCHOK expected
    '''Return class and instance keys for a class.
    '''
    if type(obj) is _Type_type:  # isclass(obj):
        return _claskey(obj, _new_style), obj
    return None, None  # not a class


def _objkey(obj):  # PYCHOK expected
    '''Return the key for any object.
    '''
    k = type(obj)
    if k is _Type_type:  # isclass(obj):
        k = _claskey(obj, _new_style)
    return k


class _NamedRef(object):
    '''Store referred object along
       with the name of the referent.
    '''
    __slots__ = ('name', 'ref')

    def __init__(self, name, ref):
        self.name = name
        self.ref = ref


# class _Slots(tuple):
#     '''Wrapper class for __slots__ attribute at class definition.
#        The instance-specific __slots__ attributes are stored in
#        a "tuple-like" space inside the instance, see Luciano
#        Ramalho, "Fluent Python", page 274+, O'Reilly, 2016 or
#        at <http://Books.Google.com/books>, then search for
#        "Fluent Python" "Space Savings with the __slots__".
#     '''
#     pass


# Kinds of _Typedefs
_i = _intern
_all_kinds = (_kind_static, _kind_dynamic, _kind_derived, _kind_ignored, _kind_inferred) = (
                _i('static'), _i('dynamic'), _i('derived'), _i('ignored'), _i('inferred'))
del _i

_Not_vari = ''  # non-variable item size


class _Typedef(object):
    '''Type definition class.
    '''
    __slots__ = {
        'base': 0,     # basic size in bytes
        'item': 0,     # item size in bytes
        'leng': None,  # or _len_...() function
        'refs': None,  # or _..._refs() function
        'both': None,  # both data and code if True, code only if False
        'kind': None,  # _kind_... value
        'type': None,  # original type
        'vari': None}  # item size attr name or _Not_vari

    def __init__(self, **kwds):
        self.reset(**kwds)

    def __lt__(self, unused):  # for Python 3+
        return True

    def __repr__(self):
        return repr(self.args())

    def __str__(self):
        t = [str(self.base), str(self.item)]
        for f in (self.leng, self.refs):
            if f:
                t.append(f.__name__)
            else:
                t.append('n/a')
        if not self.both:
            t.append('(code only)')
        return ', '.join(t)

    def args(self):  # as args tuple
        '''Return all attributes as arguments tuple.
        '''
        return (self.base, self.item, self.leng, self.refs,
                self.both, self.kind, self.type)

    def dup(self, other=None, **kwds):
        '''Duplicate attributes of dict or other typedef.
        '''
        if other is None:
            d = _dict_typedef.kwds()
        else:
            d = other.kwds()
        d.update(kwds)
        self.reset(**d)

    def flat(self, obj, mask=0):
        '''Return the aligned flat size.
        '''
        s = self.base
        if self.leng and self.item > 0:  # include items
            s += self.leng(obj) * self.item
        # workaround sys.getsizeof (and numpy?) bug ... some
        # types are incorrectly sized in some Python versions
        # (note, isinstance(obj, ()) == False)
        if not isinstance(obj, _getsizeof_excls):
            s = _getsizeof(obj, s)
        if mask:  # align
            s = (s + mask) & ~mask
        return s

    def format(self):
        '''Return format dict.
        '''
        i = self.item
        if self.vari:
            i = 'var'
        c = n = ''
        if not self.both:
            c = ' (code only)'
        if self.leng:
            n = ' (%s)' % _nameof(self.leng)
        return dict(base=self.base, item=i, leng=n, code=c,
                    kind=self.kind)

    def kwds(self):
        '''Return all attributes as keywords dict.
        '''
        return dict(base=self.base, both=self.both,
                    item=self.item, kind=self.kind,
                    leng=self.leng, refs=self.refs,
                    type=self.type, vari=self.vari)

    def save(self, t, base=0, heap=False):
        '''Save this typedef plus its class typedef.
        '''
        c, k = _keytuple(t)
        if k and k not in _typedefs:  # instance key
            _typedefs[k] = self
            if c and c not in _typedefs:  # class key
                if t.__module__ in _builtin_modules:
                    k = _kind_ignored  # default
                else:
                    k = self.kind
                _typedefs[c] = _Typedef(base=_basicsize(type(t), base=base, heap=heap),
                                        refs=_type_refs,
                                        both=False, kind=k, type=t)
        elif t not in _typedefs:
            if not _isbuiltin2(t):  # array, range, xrange in Python 2.x
                s = ' '.join((self.vari, _moduleof(t), _nameof(t)))
                s = '%r %s %s' % ((c, k), self.both, s.strip())
                raise KeyError('asizeof typedef %r bad: %s' % (self, s))

            _typedefs[t] = _Typedef(base=_basicsize(t, base=base),
                                    both=False, kind=_kind_ignored, type=t)

    def set(self, safe_len=False, **kwds):
        '''Set one or more attributes.
        '''
        if kwds:  # double check
            d = self.kwds()
            d.update(kwds)
            self.reset(**d)
        if safe_len and self.item:
            self.leng = _len

    def reset(self, base=0, item=0, leng=None, refs=None,
                    both=True, kind=None, type=None, vari=_Not_vari):
        '''Reset all specified attributes.
        '''
        if base < 0:
            raise ValueError('invalid option: %s=%r' % ('base', base))
        else:
            self.base = base
        if item < 0:
            raise ValueError('invalid option: %s=%r' % ('item', item))
        else:
            self.item = item
        if leng in _all_lens:  # XXX or _callable(leng)
            self.leng = leng
        else:
            raise ValueError('invalid option: %s=%r' % ('leng', leng))
        if refs in _all_refs:  # XXX or _callable(refs)
            self.refs = refs
        else:
            raise ValueError('invalid option: %s=%r' % ('refs', refs))
        if both in (False, True):
            self.both = both
        else:
            raise ValueError('invalid option: %s=%r' % ('both', both))
        if kind in _all_kinds:
            self.kind = kind
        else:
            raise ValueError('invalid option: %s=%r' % ('kind', kind))
        self.type = type
        self.vari = vari or _Not_vari
        if str(self.vari) != self.vari:
            raise ValueError('invalid option: %s=%r' % ('vari', vari))


_typedefs = {}  # type: Dict[type, _Typedef]


def _typedef_both(t, base=0, item=0, leng=None, refs=None,
                     kind=_kind_static, heap=False, vari=_Not_vari):
    '''Add new typedef for both data and code.
    '''
    v = _Typedef(base=_basicsize(t, base=base), item=_itemsize(t, item),
                 refs=refs, leng=leng,
                 both=True, kind=kind, type=t, vari=vari)
    v.save(t, base=base, heap=heap)
    return v  # for _dict_typedef


def _typedef_code(t, base=0, refs=None, kind=_kind_static, heap=False):
    '''Add new typedef for code only.
    '''
    v = _Typedef(base=_basicsize(t, base=base),
                 refs=refs,
                 both=False, kind=kind, type=t)
    v.save(t, base=base, heap=heap)
    return v  # for _dict_typedef


# Static typedefs for data and code types
_typedef_both(complex)
_typedef_both(float)
_typedef_both(list, refs=_seq_refs, leng=_len_list, item=_sizeof_Cvoidp)  # sizeof(PyObject*)
_typedef_both(tuple, refs=_seq_refs, leng=_len, item=_sizeof_Cvoidp)  # sizeof(PyObject*)
_typedef_both(property, refs=_prop_refs)
_typedef_both(type(Ellipsis))
_typedef_both(type(None))

# _Slots are "tuple-like", REMOVED see _Slots.__doc__
# _typedef_both(_Slots, item=_sizeof_Cvoidp,
#               leng=_len_slots,  # length less one
#               refs=None,  # but no referents
#               heap=True)  # plus head

# dict, dictproxy, dict_proxy and other dict-like types
_dict_typedef = _typedef_both(dict, item=_sizeof_CPyDictEntry, leng=_len_dict, refs=_dict_refs)
# XXX any class __dict__ is <type dict_proxy> in Python 3+?
_typedef_both(type(_Typedef.__dict__), item=_sizeof_CPyDictEntry, leng=_len_dict, refs=_dict_refs)
# other dict-like classes and types may be derived or inferred,
# provided the module and class name is listed here (see functions
# adict, _isdictclass and _infer_dict for further details)
_dict_classes = {'UserDict': ('IterableUserDict', 'UserDict'),
                 'weakref': ('WeakKeyDictionary', 'WeakValueDictionary')}
try:  # <type module> is essentially a dict
    _typedef_both(Types.ModuleType, base=_dict_typedef.base,
                  item=_dict_typedef.item + _sizeof_CPyModuleObject,
                  leng=_len_module, refs=_module_refs)
except AttributeError:  # missing
    pass

# Newer or obsolete types
from array import array  # array type

def _array_kwds(obj):
    if hasattr(obj, 'itemsize'):
        v = 'itemsize'
    else:
        v = _Not_vari
    # since item size varies by the array data type, set
    # itemsize to 1 byte and use _len_array in bytes; note,
    # function itemsize returns the actual size in bytes
    # and function leng returns the length in number of items
    return dict(leng=_len_array, item=_sizeof_Cbyte, vari=v)

def _len_array(obj):
    '''Array length (in bytes!).
    '''
    return len(obj) * obj.itemsize

_all_lens += (_len_array,)  # type: ignore

_typedef_both(array, **_array_kwds(array('d', [])))

v = sys.version_info
_array_excl = (v[0] == 2 and v < (2, 7, 4)) or \
              (v[0] == 3 and v < (3, 2, 4))
if _array_excl:  # see function _typedef below
    _getsizeof_excls_add(array)

del v

try:  # bool has non-zero __itemsize__ in 3.0
    _typedef_both(bool)
except NameError:  # missing
    pass

try:
    _typedef_both(bytearray, item=_sizeof_Cbyte, leng=_len_bytearray)
except NameError:  # bytearray new in 2.6, 3.0
    pass
try:
    if type(bytes) is not type(str):  # bytes is str in 2.6, bytes new in 2.6, 3.0
        _typedef_both(bytes, item=_sizeof_Cbyte, leng=_len)  # bytes new in 2.6, 3.0
except NameError:  # missing
    pass
# try:  # XXX like bytes
#     _typedef_both(str8, item=_sizeof_Cbyte, leng=_len)  # str8 new in 2.6, 3.0
# except NameError:  # missing
#     pass

try:
    _typedef_both(enumerate, refs=_enum_refs)
except NameError:  # missing
    pass

try:  # Exception is type in Python 3+
    _typedef_both(Exception, refs=_exc_refs)
except Exception:  # missing
    pass

try:
    _typedef_both(frozenset, item=_sizeof_Csetentry, leng=_len_set, refs=_seq_refs)
except NameError:  # missing
    pass
try:
    _typedef_both(set, item=_sizeof_Csetentry, leng=_len_set, refs=_seq_refs)
except NameError:  # missing
    pass

try:  # not callable()
    _typedef_both(Types.GetSetDescriptorType)
except AttributeError:  # missing
    pass

_typedef_both(int, item=_sizeof_Cdigit, leng=_len_int)

try:  # not callable()
    _typedef_both(Types.MemberDescriptorType)
except AttributeError:  # missing
    pass

try:
    _typedef_both(type(NotImplemented))  # == Types.NotImplementedType
except NameError:  # missing
    pass

try:  # MCCABE 14
    import numpy  # NumPy array, matrix, etc.

    def _isnumpy(obj):
        '''Return True for a NumPy arange, array, matrix, etc. instance.
        '''
        try:
            return isinstance(obj, _numpy_types) or (hasattr(obj, 'nbytes') and
                                          _moduleof(_classof(obj)).startswith('numpy'))
        except (AttributeError, OSError, ValueError):  # on iOS/Pythonista
            return False

    def _len_numpy(obj):
        '''NumPy array, matrix, etc. length (in bytes!).
        '''
        return obj.nbytes  # == obj.size * obj.itemsize

    def _numpy_kwds(obj):
        b = _getsizeof(obj, 96) - obj.nbytes  # XXX 96..144 typical?
        # since item size depends on the numpy data type, set
        # itemsize to 1 byte and use _len_numpy in bytes; note,
        # function itemsize returns the actual size in bytes,
        # function alen returns the length in number of items
        return dict(base=b, item=_sizeof_Cbyte,  # not obj.itemsize
                            leng=_len_numpy,
                            refs=_numpy_refs,
                            vari='itemsize')

    def _numpy_refs(obj, named):
        '''Return the .base object for NumPy slices, views, etc.
        '''
        return _refs(obj, named, 'base')

    _all_lens += (_len_numpy,)
    _all_refs += (_numpy_refs,)

    v = tuple(map(int, numpy.__version__.split('.')[:2]))

    if v < (1, 19):
        t = (numpy.matrix(range(0)),)
    else:  # numpy.matrix deprecated in 1.19.3
        t = ()
    _numpy_types = ()  # type: Tuple[type, ...]
    for d in (t + (numpy.array(range(0)), numpy.arange(0),
              numpy.ma.masked_array([]), numpy.ndarray(0))):
        t = type(d)
        if t not in _numpy_types:
            _numpy_types += (t,)
            if _isnumpy(d):  # double check
                _typedef_both(t, **_numpy_kwds(d))
            else:
                raise AssertionError('not %s: %r' % ('numpy', d))

    # sizing numpy 1.13 arrays works fine, but 1.8 and older
    # appears to suffer from sys.getsizeof() bug like array
    _numpy_excl = v < (1, 9)
    if _numpy_excl:  # see function _typedef below
        for t in _numpy_types:
            _getsizeof_excls_add(t)

    del d, t, v
except ImportError:  # no NumPy
    _numpy_excl = numpy = None  # type: ignore # see function _typedef below

    def _isnumpy(obj):  # PYCHOK expected
        '''Not applicable, no NumPy.
        '''
        return False

try:
    _typedef_both(range)
except NameError:  # missing
    pass

try:
    _typedef_both(reversed, refs=_enum_refs)
except NameError:  # missing
    pass

try:
    _typedef_both(slice, item=_sizeof_Cvoidp, leng=_len_slice)  # XXX worst-case itemsize?
except NameError:  # missing
    pass

try:
    from os import stat
    _typedef_both(type(stat(curdir)), refs=_stat_refs)  # stat_result
except ImportError:  # missing
    pass

try:
    from os import statvfs
    _typedef_both(type(statvfs(curdir)), refs=_statvfs_refs,  # statvfs_result
                  item=_sizeof_Cvoidp, leng=_len)
except ImportError:  # missing
    pass

try:
    from struct import Struct  # only in Python 2.5 and 3.0
    _typedef_both(Struct, item=_sizeof_Cbyte, leng=_len_struct)  # len in bytes
except ImportError:  # missing
    pass

try:
    _typedef_both(Types.TracebackType, refs=_tb_refs)
except AttributeError:  # missing
    pass

_typedef_both(str, leng=_len_unicode, item=_sizeof_Cunicode)

try:  # <type 'KeyedRef'>
    _typedef_both(Weakref.KeyedRef, refs=_weak_refs, heap=True)  # plus head
except AttributeError:  # missing
    pass

try:  # <type 'weakproxy'>
    _typedef_both(Weakref.ProxyType)
except AttributeError:  # missing
    pass

try:  # <type 'weakref'>
    _typedef_both(Weakref.ReferenceType, refs=_weak_refs)
except AttributeError:  # missing
    pass

# some other, callable types
_typedef_code(object, kind=_kind_ignored)
_typedef_code(super, kind=_kind_ignored)
_typedef_code(_Type_type, kind=_kind_ignored)

try:
    _typedef_code(classmethod, refs=_im_refs)
except NameError:
    pass
try:
    _typedef_code(staticmethod, refs=_im_refs)
except NameError:
    pass
try:
    _typedef_code(Types.MethodType, refs=_im_refs)
except NameError:
    pass

try:  # generator (expression), no itemsize, no len(), not callable()
    _typedef_both(Types.GeneratorType, refs=_gen_refs)
except AttributeError:  # missing
    pass

try:  # <type 'weakcallableproxy'>
    _typedef_code(Weakref.CallableProxyType, refs=_weak_refs)
except AttributeError:  # missing
    pass

# any type-specific iterators
s = [_items({}), _keys({}), _values({})]
try:  # reversed list and tuples iterators
    s.extend([reversed([]), reversed(())])
except NameError:  # missing
    pass

try:  # callable-iterator
    from re import finditer
    s.append(finditer('', ''))
except ImportError:  # missing
    pass

for t in _values(_typedefs):
    if t.type and t.leng:
        try:  # create an (empty) instance
            s.append(t.type())
        except TypeError:
            pass
for t in s:
    try:
        i = iter(t)
        _typedef_both(type(i), leng=_len_iter, refs=_iter_refs, item=0)  # no itemsize!
    except (KeyError, TypeError):  # ignore non-iterables, duplicates, etc.
        pass
del i, s, t


def _typedef(obj, derive=False, frames=False, infer=False):  # MCCABE 25
    '''Create a new typedef for an object.
    '''
    t = type(obj)
    v = _Typedef(base=_basicsize(t, obj=obj),
                 kind=_kind_dynamic, type=t)
#   _printf('new %r %r/%r %s', t, _basicsize(t), _itemsize(t), _repr(dir(obj)))
    if ismodule(obj):  # handle module like dict
        v.dup(item=_dict_typedef.item + _sizeof_CPyModuleObject,
              leng=_len_module,
              refs=_module_refs)
    elif _isframe(obj):
        v.set(base=_basicsize(t, base=_sizeof_CPyFrameObject, obj=obj),
              item=_itemsize(t),
              leng=_len_frame,
              refs=_frame_refs)
        if not frames:  # ignore frames
            v.set(kind=_kind_ignored)
    elif iscode(obj):
        v.set(base=_basicsize(t, base=_sizeof_CPyCodeObject, obj=obj),
              item=_sizeof_Cvoidp,
              leng=_len_code,
              refs=_co_refs,
              both=False)  # code only
    elif _callable(obj):
        if isclass(obj):  # class or type
            v.set(refs=_class_refs,
                  both=False)  # code only
            if _moduleof(obj) in _builtin_modules:
                v.set(kind=_kind_ignored)
        elif isbuiltin(obj):  # function or method
            v.set(both=False,  # code only
                  kind=_kind_ignored)
        elif isfunction(obj):
            v.set(refs=_func_refs,
                  both=False)  # code only
        elif ismethod(obj):
            v.set(refs=_im_refs,
                  both=False)  # code only
        elif isclass(t):  # callable instance, e.g. SCons,
            # handle like any other instance further below
            v.set(item=_itemsize(t), safe_len=True,
                  refs=_inst_refs)  # not code only!
        else:
            v.set(both=False)  # code only
    elif _issubclass(t, dict):
        v.dup(kind=_kind_derived)
    elif _isdictclass(obj) or (infer and _infer_dict(obj)):
        v.dup(kind=_kind_inferred)
    elif _iscell(obj):
        v.set(item=_itemsize(t), refs=_cell_refs)
    elif _isnamedtuple(obj):
        v.set(refs=_namedtuple_refs)
    elif numpy and _isnumpy(obj):  # NumPy data
        v.set(**_numpy_kwds(obj))
        if _numpy_excl:
            _getsizeof_excls_add(t)
    elif array and isinstance(obj, array):
        v.set(**_array_kwds(obj))
        if _array_excl:
            _getsizeof_excls_add(t)
    elif _moduleof(obj) in _builtin_modules:
        v.set(kind=_kind_ignored)
    else:  # assume an instance of some class
        if derive:
            p = _derive_typedef(t)
            if p:  # duplicate parent
                v.dup(other=p, kind=_kind_derived)
                return v
        if _issubclass(t, Exception):
            v.set(item=_itemsize(t), safe_len=True,
                  refs=_exc_refs,
                  kind=_kind_derived)
        elif isinstance(obj, Exception):
            v.set(item=_itemsize(t), safe_len=True,
                  refs=_exc_refs)
        else:
            v.set(item=_itemsize(t), safe_len=True,
                  refs=_inst_refs)
    return v


class _Prof(object):
    '''Internal type profile class.
    '''
    high   = 0      # largest size
    number = 0      # number of (unique) objects
    objref = None   # largest obj (weakref)
    total  = 0      # total size
    weak   = False  # objref is weakref(obj)

    def __cmp__(self, other):
        if self.total < other.total:
            return -1
        elif self.total > other.total:
            return +1
        elif self.number < other.number:
            return -1
        elif self.number > other.number:
            return +1
        return 0

    def __lt__(self, other):  # for Python 3+
        return self.__cmp__(other) < 0

    def format(self, clip=0, grand=None):
        '''Return format dict.
        '''
        if self.number > 1:  # avg., plural
            a, p = int(self.total / self.number), 's'
        else:
            a, p = self.total, ''
        o = self.objref
        if self.weak:
            o = o()
        t = _SI2(self.total)
        if grand:
            t += ' (%s)' % _p100(self.total, grand, prec=0)
        return dict(avg=_SI2(a), high=_SI2(self.high),
                    lengstr=_lengstr(o), obj=_repr(o, clip=clip),
                    plural=p, total=t)

    def update(self, obj, size):
        '''Update this profile.
        '''
        self.number += 1
        self.total += size
        if self.high < size:  # largest
            self.high = size
            try:  # prefer using weak ref
                self.objref, self.weak = Weakref.ref(obj), True
            except TypeError:
                self.objref, self.weak = obj, False


class _Rank(object):
    '''Internal largest object class.
    '''
    __slots__ = {
        'deep':   0,      # recursion depth
        'id':     0,      # obj id
        'key':    None,   # Typedef
        'objref': None,   # obj or Weakref.ref(obj)
        'pid':    0,      # parent obj id
        'size':   0,      # size in bytes
        'weak':   False}  # objref is Weakref.ref

    def __init__(self, key, obj, size, deep, pid):
        self.deep = deep
        self.id = id(obj)
        self.key = key
        try:  # prefer using weak ref
            self.objref, self.weak = Weakref.ref(obj), True
        except TypeError:
            self.objref, self.weak = obj, False
        self.pid = pid
        self.size = size

    def format(self, clip=0, id2x={}):
        '''Return string.
        '''
        o = self.objref
        if self.weak:
            o = o()
        if self.deep > 0:
            d = ' (at %s)' % (self.deep,)
        else:
            d = ''
        if self.pid:
            p = ', pix %s' % (id2x.get(self.pid, '?'),)
        else:
            p = ''
        return '%s: %s%s, ix %d%s%s' % (_prepr(self.key, clip=clip),
               _repr(o, clip=clip), _lengstr(o), id2x[self.id], d, p)


class _Seen(dict):
    '''Internal obj visits counter.
    '''
    def again(self, key):
        try:
            s = self[key] + 1
        except KeyError:
            s = 1
        if s > 0:
            self[key] = s


# Public classes

class Asized(object):
    '''Stores the results of an **asized** object in the following
       4 attributes:

        *size* -- total size of the object (including referents)

        *flat* -- flat size of the object (in bytes)

        *name* -- name or ``repr`` of the object

        *refs* -- tuple containing an **Asized** instance for each referent
    '''
    __slots__ = ('flat', 'name', 'refs', 'size')

    def __init__(self, size, flat, refs=(), name=None):
        self.size = size  # total size
        self.flat = flat  # flat size
        self.name = name  # name, repr or None
        self.refs = tuple(refs)

    def __str__(self):
        return 'size %r, flat %r, refs[%d], name %r' % (
            self.size, self.flat, len(self.refs), self.name)

    def format(self, format='%(name)s size=%(size)d flat=%(flat)d',
                     depth=-1, order_by='size', indent=''):
        '''Format the size information of the object and of all
           sized referents as a string.

            *format* -- Specifies the format per instance (with 'name',
                        'size' and 'flat' as interpolation parameters)

            *depth* -- Recursion level up to which the referents are
                       printed (use -1 for unlimited)

            *order_by* -- Control sort order of referents, valid choices
                          are 'name', 'size' and 'flat'

            *indent* -- Optional indentation (default '')
        '''
        t = indent + (format % dict(size=self.size, flat=self.flat,
                                    name=self.name))
        if depth and self.refs:
            rs = sorted(self.refs, key=lambda x: getattr(x, order_by),
                                   reverse=order_by in ('size', 'flat'))
            rs = [r.format(format=format, depth=depth-1, order_by=order_by,
                           indent=indent+'    ') for r in rs]
            t = '\n'.join([t] + rs)
        return t

    def get(self, name, dflt=None):
        '''Return the named referent (or *dflt* if not found).
        '''
        for ref in self.refs:
            if name == ref.name:
                return ref
        return dflt


class Asizer(object):
    '''Sizer state and options to accumulate sizes.
    '''
    _above_  = 1024   # rank only objs of size 1K+
    _align_  = 8
    _clip_   = 80
    _code_   = False
    _cutoff_ = 0  # in percent
    _derive_ = False
    _detail_ = 0  # for Asized only
    _frames_ = False
    _infer_  = False
    _limit_  = 100
    _stats_  = 0

    _depth   = 0  # deepest recursion
    _excl_d  = None  # {}
    _ign_d   = _kind_ignored
    _incl    = ''  # or ' (incl. code)'
    _mask    = 7   # see _align_
    _missed  = 0   # due to errors
    _profile = False  # no profiling
    _profs   = None   # {}
    _ranked  = 0
    _ranks   = []     # type: List[_Rank] # sorted by decreasing size
    _seen    = None   # {}
    _stream  = None   # I/O stream for printing
    _total   = 0      # total size

    def __init__(self, **opts):
        '''New **Asizer** accumulator.

           See this module documentation for more details.
           See method **reset** for all available options and defaults.
        '''
        self._excl_d = {}
        self.reset(**opts)

    def _c100(self, stats):
        '''Cutoff as percentage (for backward compatibility)
        '''
        s = int(stats)
        c = int((stats - s) * 100.0 + 0.5) or self.cutoff
        return s, c

    def _clear(self):
        '''Clear state.
        '''
        self._depth = 0   # recursion depth reached
        self._incl = ''  # or ' (incl. code)'
        self._missed = 0   # due to errors
        self._profile = False
        self._profs = {}
        self._ranked = 0
        self._ranks = []
        self._seen = _Seen()
        self._total = 0   # total size
        for k in _keys(self._excl_d):
            self._excl_d[k] = 0
        # don't size, profile or rank private, possibly large objs
        m = sys.modules[__name__]
        self.exclude_objs(self, self._excl_d, self._profs, self._ranks,
                                self._seen, m, m.__dict__, m.__doc__,
                               _typedefs)

    def _nameof(self, obj):
        '''Return the object's name.
        '''
        return _nameof(obj, '') or self._repr(obj)

    def _prepr(self, obj):
        '''Like **prepr()**.
        '''
        return _prepr(obj, clip=self._clip_)

    def _printf(self, fmt, *args, **print3options):
        '''Print to sys.stdout or the configured stream if any is
           specified and if the file keyword argument is not already
           set in the **print3options** for this specific call.
        '''
        if self._stream and not print3options.get('file', None):
            if args:
                fmt = fmt % args
            _printf(fmt, file=self._stream, **print3options)
        else:
            _printf(fmt, *args, **print3options)

    def _prof(self, key):
        '''Get _Prof object.
        '''
        p = self._profs.get(key, None)
        if not p:
            self._profs[key] = p = _Prof()
            self.exclude_objs(p)  # XXX superfluous?
        return p

    def _rank(self, key, obj, size, deep, pid):
        '''Rank 100 largest objects by size.
        '''
        rs = self._ranks
        # bisect, see <http://GitHub.com/python/cpython/blob/master/Lib/bisect.py>
        i, j = 0, len(rs)
        while i < j:
            m = (i + j) // 2
            if size < rs[m].size:
                i = m + 1
            else:
                j = m
        if i < 100:
            r = _Rank(key, obj, size, deep, pid)
            rs.insert(i, r)
            self.exclude_objs(r)  # XXX superfluous?
            while len(rs) > 100:
                rs.pop()
            # self._ranks[:] = rs[:100]
        self._ranked += 1

    def _repr(self, obj):
        '''Like ``repr()``.
        '''
        return _repr(obj, clip=self._clip_)

    def _sizer(self, obj, pid, deep, sized):  # MCCABE 19
        '''Size an object, recursively.
        '''
        s, f, i = 0, 0, id(obj)
        if i not in self._seen:
            self._seen[i] = 1
        elif deep or self._seen[i]:
            # skip obj if seen before
            # or if ref of a given obj
            self._seen.again(i)
            if sized:
                s = sized(s, f, name=self._nameof(obj))
                self.exclude_objs(s)
            return s  # zero
        else:  # deep == seen[i] == 0
            self._seen.again(i)
        try:
            k, rs = _objkey(obj), []
            if k in self._excl_d:
                self._excl_d[k] += 1
            else:
                v = _typedefs.get(k, None)
                if not v:  # new typedef
                    _typedefs[k] = v = _typedef(obj, derive=self._derive_,
                                                     frames=self._frames_,
                                                      infer=self._infer_)
                if (v.both or self._code_) and v.kind is not self._ign_d:
                    s = f = v.flat(obj, self._mask)  # flat size
                    if self._profile:
                        # profile based on *flat* size
                        self._prof(k).update(obj, s)
                    # recurse, but not for nested modules
                    if v.refs and deep < self._limit_ \
                              and not (deep and ismodule(obj)):
                        # add sizes of referents
                        z, d = self._sizer, deep + 1
                        if sized and deep < self._detail_:
                            # use named referents
                            self.exclude_objs(rs)
                            for o in v.refs(obj, True):
                                if isinstance(o, _NamedRef):
                                    r = z(o.ref, i, d, sized)
                                    r.name = o.name
                                else:
                                    r = z(o, i, d, sized)
                                    r.name = self._nameof(o)
                                rs.append(r)
                                s += r.size
                        else:  # just size and accumulate
                            for o in v.refs(obj, False):
                                s += z(o, i, d, None)
                        # deepest recursion reached
                        if self._depth < d:
                            self._depth = d
                if self._stats_ and s > self._above_ > 0:
                    # rank based on *total* size
                    self._rank(k, obj, s, deep, pid)
        except RuntimeError:  # XXX RecursionLimitExceeded:
            self._missed += 1
        if not deep:
            self._total += s  # accumulate
        if sized:
            s = sized(s, f, name=self._nameof(obj), refs=rs)
            self.exclude_objs(s)
        return s

    def _sizes(self, objs, sized=None):
        '''Return the size or an **Asized** instance for each
           given object plus the total size.  The total includes
           the size of duplicates only once.
        '''
        self.exclude_refs(*objs)  # skip refs to objs
        s, t = {}, []
        self.exclude_objs(s, t)
        for o in objs:
            i = id(o)
            if i in s:  # duplicate
                self._seen.again(i)
            else:
                s[i] = self._sizer(o, 0, 0, sized)
            t.append(s[i])
        return tuple(t)

    @property
    def above(self):
        '''Get the large object size threshold (int).
        '''
        return self._above_

    @property
    def align(self):
        '''Get the size alignment (int).
        '''
        return self._align_

    def asized(self, *objs, **opts):
        '''Size each object and return an **Asized** instance with
           size information and referents up to the given detail
           level (and with modified options, see method **set**).

           If only one object is given, the return value is the
           **Asized** instance for that object.  The **Asized** size
           of duplicate and ignored objects will be zero.
        '''
        if opts:
            self.set(**opts)
        t = self._sizes(objs, Asized)
        if len(t) == 1:
            t = t[0]
        return t

    def asizeof(self, *objs, **opts):
        '''Return the combined size of the given objects
           (with modified options, see method **set**).
        '''
        if opts:
            self.set(**opts)
        self.exclude_refs(*objs)  # skip refs to objs
        return sum(self._sizer(o, 0, 0, None) for o in objs)

    def asizesof(self, *objs, **opts):
        '''Return the individual sizes of the given objects
           (with modified options, see method  **set**).

           The size of duplicate and ignored objects will be zero.
        '''
        if opts:
            self.set(**opts)
        return self._sizes(objs, None)

    @property
    def clip(self):
        '''Get the clipped string length (int).
        '''
        return self._clip_

    @property
    def code(self):
        '''Size (byte) code (bool).
        '''
        return self._code_

    @property
    def cutoff(self):
        '''Stats cutoff (int).
        '''
        return self._cutoff_

    @property
    def derive(self):
        '''Derive types (bool).
        '''
        return self._derive_

    @property
    def detail(self):
        '''Get the detail level for **Asized** refs (int).
        '''
        return self._detail_

    @property
    def duplicate(self):
        '''Get the number of duplicate objects seen so far (int).
        '''
        return sum(1 for v in _values(self._seen) if v > 1)  # == len

    def exclude_objs(self, *objs):
        '''Exclude the specified objects from sizing, profiling and ranking.
        '''
        for o in objs:
            self._seen.setdefault(id(o), -1)

    def exclude_refs(self, *objs):
        '''Exclude any references to the specified objects from sizing.

           While any references to the given objects are excluded, the
           objects will be sized if specified as positional arguments
           in subsequent calls to methods **asizeof** and **asizesof**.
        '''
        for o in objs:
            self._seen.setdefault(id(o), 0)

    def exclude_types(self, *objs):
        '''Exclude the specified object instances and types from sizing.

           All instances and types of the given objects are excluded,
           even objects specified as positional arguments in subsequent
           calls to methods **asizeof** and **asizesof**.
        '''
        for o in objs:
            for t in _keytuple(o):
                if t and t not in self._excl_d:
                    self._excl_d[t] = 0

    @property
    def excluded(self):
        '''Get the types being excluded (tuple).
        '''
        return tuple(_keys(self._excl_d))

    @property
    def frames(self):
        '''Ignore stack frames (bool).
        '''
        return self._frames_

    @property
    def ignored(self):
        '''Ignore certain types (bool).
        '''
        return True if self._ign_d else False

    @property
    def infer(self):
        '''Infer types (bool).
        '''
        return self._infer_

    @property
    def limit(self):
        '''Get the recursion limit (int).
        '''
        return self._limit_

    @property
    def missed(self):
        '''Get the number of objects missed due to errors (int).
        '''
        return self._missed

    def print_largest(self, w=0, cutoff=0, **print3options):
        '''Print the largest objects.

           The available options and defaults are:

            *w=0*           -- indentation for each line

            *cutoff=100*    -- number of largest objects to print

            *print3options* -- some keyword arguments, like Python 3+ print
        '''
        c = int(cutoff) if cutoff else self._cutoff_
        n = min(len(self._ranks), max(c, 0))
        s = self._above_
        if n > 0 and s > 0:
            self._printf('%s%*d largest object%s (of %d over %d bytes%s)', linesep,
                          w, n, _plural(n), self._ranked, s, _SI(s), **print3options)
            id2x = dict((r.id, i) for i, r in enumerate(self._ranks))
            for r in self._ranks[:n]:
                s, t = r.size, r.format(self._clip_, id2x)
                self._printf('%*d bytes%s: %s', w, s, _SI(s), t, **print3options)

    def print_profiles(self, w=0, cutoff=0, **print3options):
        '''Print the profiles above *cutoff* percentage.

           The available options and defaults are:

                *w=0*           -- indentation for each line

                *cutoff=0*      -- minimum percentage printed

                *print3options* -- some keyword arguments, like Python 3+ print
        '''
        # get the profiles with non-zero size or count
        t = [(v, k) for k, v in _items(self._profs) if v.total > 0 or v.number > 1]
        if (len(self._profs) - len(t)) < 9:  # just show all
            t = [(v, k) for k, v in _items(self._profs)]
        if t:
            s = ''
            if self._total:
                s = ' (% of grand total)'
                c = int(cutoff) if cutoff else self._cutoff_
                C = int(c * 0.01 * self._total)
            else:
                C = c = 0
            self._printf('%s%*d profile%s:  total%s, average, and largest flat size%s:  largest object',
                         linesep, w, len(t), _plural(len(t)), s, self._incl, **print3options)
            r = len(t)
            for v, k in sorted(t, reverse=True):
                s = 'object%(plural)s:  %(total)s, %(avg)s, %(high)s:  %(obj)s%(lengstr)s' % v.format(self._clip_, self._total)
                self._printf('%*d %s %s', w, v.number, self._prepr(k), s, **print3options)
                r -= 1
                if r > 1 and v.total < C:
                    self._printf('%+*d profiles below cutoff (%.0f%%)', w, r, c)
                    break
            z = len(self._profs) - len(t)
            if z > 0:
                self._printf('%+*d %r object%s', w, z, 'zero', _plural(z), **print3options)

    def print_stats(self, objs=(), opts={}, sized=(), sizes=(), stats=3, **print3options):
        '''Prints the statistics.

           The available options and defaults are:

                *w=0*           -- indentation for each line

                *objs=()*       -- optional, list of objects

                *opts={}*       -- optional, dict of options used

                *sized=()*      -- optional, tuple of **Asized** instances returned

                *sizes=()*      -- optional, tuple of sizes returned

                *stats=3*       -- print stats, see function **asizeof**

                *print3options* -- some keyword arguments, like Python 3+ print
        '''
        s = min(opts.get('stats', stats) or 0, self.stats)
        if s > 0:  # print stats
            w = len(str(self.missed + self.seen + self.total)) + 1
            t = c = ''
            o = _kwdstr(**opts)
            if o and objs:
                c = ', '
            # print header line(s)
            if sized and objs:
                n = len(objs)
                if n > 1:
                    self._printf('%sasized(...%s%s) ...', linesep, c, o, **print3options)
                    for i in range(n):  # no enumerate in Python 2.2.3
                        self._printf('%*d: %s', w - 1, i, sized[i], **print3options)
                else:
                    self._printf('%sasized(%s): %s', linesep, o, sized, **print3options)
            elif sizes and objs:
                self._printf('%sasizesof(...%s%s) ...', linesep, c, o, **print3options)
                for z, o in zip(sizes, objs):
                    self._printf('%*d bytes%s%s:  %s', w, z, _SI(z), self._incl, self._repr(o), **print3options)
            else:
                if objs:
                    t = self._repr(objs)
                self._printf('%sasizeof(%s%s%s) ...', linesep, t, c, o, **print3options)
            # print summary
            self.print_summary(w=w, objs=objs, **print3options)
            # for backward compatibility, cutoff from fractional stats
            s, c = self._c100(s)
            self.print_largest(w=w, cutoff=c if s < 2 else 10, **print3options)
            if s > 1:  # print profile
                self.print_profiles(w=w, cutoff=c, **print3options)
                if s > 2:  # print typedefs
                    self.print_typedefs(w=w, **print3options)  # PYCHOK .print_largest?

    def print_summary(self, w=0, objs=(), **print3options):
        '''Print the summary statistics.

           The available options and defaults are:

                *w=0*           -- indentation for each line

                *objs=()*       -- optional, list of objects

                *print3options* -- some keyword arguments, like Python 3+ print
        '''
        self._printf('%*d bytes%s%s', w, self._total, _SI(self._total), self._incl, **print3options)
        if self._mask:
            self._printf('%*d byte aligned', w, self._mask + 1, **print3options)
        self._printf('%*d byte sizeof(void*)', w, _sizeof_Cvoidp, **print3options)
        n = len(objs or ())
        self._printf('%*d object%s %s', w, n, _plural(n), 'given', **print3options)
        n = self.sized
        self._printf('%*d object%s %s', w, n, _plural(n), 'sized', **print3options)
        if self._excl_d:
            n = sum(_values(self._excl_d))
            self._printf('%*d object%s %s', w, n, _plural(n), 'excluded', **print3options)
        n = self.seen
        self._printf('%*d object%s %s', w, n, _plural(n), 'seen', **print3options)
        n = self.ranked
        if n > 0:
            self._printf('%*d object%s %s', w, n, _plural(n), 'ranked', **print3options)
        n = self.missed
        self._printf('%*d object%s %s', w, n, _plural(n), 'missed', **print3options)
        n = self.duplicate
        self._printf('%*d duplicate%s', w, n, _plural(n), **print3options)
        if self._depth > 0:
            self._printf('%*d deepest recursion', w, self._depth, **print3options)

    def print_typedefs(self, w=0, **print3options):
        '''Print the types and dict tables.

           The available options and defaults are:

                *w=0*           -- indentation for each line

                *print3options* -- some keyword arguments, like Python 3+ print
        '''
        for k in _all_kinds:
            # XXX Python 3+ doesn't sort type objects
            t = [(self._prepr(a), v) for a, v in _items(_typedefs)
                                      if v.kind == k and (v.both or self._code_)]
            if t:
                self._printf('%s%*d %s type%s:  basicsize, itemsize, _len_(), _refs()',
                             linesep, w, len(t), k, _plural(len(t)), **print3options)
                for a, v in sorted(t):
                    self._printf('%*s %s:  %s', w, '', a, v, **print3options)
        # dict and dict-like classes
        t = sum(len(v) for v in _values(_dict_classes))
        if t:
            self._printf('%s%*d dict/-like classes:', linesep, w, t, **print3options)
            for m, v in _items(_dict_classes):
                self._printf('%*s %s:  %s', w, '', m, self._prepr(v), **print3options)

    @property
    def ranked(self):
        '''Get the number objects ranked by size so far (int).
        '''
        return self._ranked

    def reset(self, above=1024, align=8, clip=80, code=False,  # PYCHOK too many args
                    cutoff=10, derive=False, detail=0, frames=False, ignored=True,
                    infer=False, limit=100, stats=0, stream=None, **extra):
        '''Reset sizing options, state, etc. to defaults.

           The available options and default values are:

                *above=0*      -- threshold for largest objects stats

                *align=8*      -- size alignment

                *code=False*   -- incl. (byte)code size

                *cutoff=10*    -- limit large objects or profiles stats

                *derive=False* -- derive from super type

                *detail=0*     -- **Asized** refs level

                *frames=False* -- ignore frame objects

                *ignored=True* -- ignore certain types

                *infer=False*  -- try to infer types

                *limit=100*    -- recursion limit

                *stats=0*      -- print statistics, see function **asizeof**

                *stream=None*  -- output stream for printing

           See function **asizeof** for a description of the options.
        '''
        if extra:
            t = _plural(len(extra)), _kwdstr(**extra)
            raise KeyError('invalid option%s: %s' % t)
        # options
        self._above_ = above
        self._align_ = align
        self._clip_ = clip
        self._code_ = code
        self._cutoff_ = cutoff
        self._derive_ = derive
        self._detail_ = detail  # for Asized only
        self._frames_ = frames
        self._infer_ = infer
        self._limit_ = limit
        self._stats_ = stats
        self._stream = stream
        if ignored:
            self._ign_d = _kind_ignored
        else:
            self._ign_d = None
        # clear state
        self._clear()
        self.set(align=align, code=code, cutoff=cutoff, stats=stats)

    @property
    def seen(self):
        '''Get the number objects seen so far (int).
        '''
        return sum(v for v in _values(self._seen) if v > 0)

    def set(self, above=None, align=None, code=None, cutoff=None,
                  frames=None, detail=None, limit=None, stats=None):
        '''Set some sizing options.  See also **reset**.

           The available options are:

                *above*  -- threshold for largest objects stats

                *align*  -- size alignment

                *code*   -- incl. (byte)code size

                *cutoff* -- limit large objects or profiles stats

                *detail* -- **Asized** refs level

                *frames* -- size or ignore frame objects

                *limit*  -- recursion limit

                *stats*  -- print statistics, see function **asizeof**

           Any options not set remain unchanged from the previous setting.
        '''
        # adjust
        if above is not None:
            self._above_ = int(above)
        if align is not None:
            self._align_ = align
            if align > 1:
                self._mask = align - 1
                if (self._mask & align) != 0:
                    raise ValueError('invalid option: %s=%r' % ('align', align))
            else:
                self._mask = 0
        if code is not None:
            self._code_ = code
            if code:  # incl. (byte)code
                self._incl = ' (incl. code)'
        if detail is not None:
            self._detail_ = detail
        if frames is not None:
            self._frames_ = frames
        if limit is not None:
            self._limit_ = limit
        if stats is not None:
            if stats < 0:
                raise ValueError('invalid option: %s=%r' % ('stats', stats))
            # for backward compatibility, cutoff from fractional stats
            s, c = self._c100(stats)
            self._cutoff_ = int(cutoff) if cutoff else c
            self._stats_ = s
            self._profile = s > 1  # profile types

    @property
    def sized(self):
        '''Get the number objects sized so far (int).
        '''
        return sum(1 for v in _values(self._seen) if v > 0)

    @property
    def stats(self):
        '''Get the stats and cutoff setting (float).
        '''
        return self._stats_  # + (self._cutoff_ * 0.01)

    @property
    def total(self):
        '''Get the total size (in bytes) accumulated so far.
        '''
        return self._total


# Public functions

def adict(*classes):
    '''Install one or more classes to be handled as dict.
    '''
    a = True
    for c in classes:
        # if class is dict-like, add class
        # name to _dict_classes[module]
        if isclass(c) and _infer_dict(c):
            t = _dict_classes.get(c.__module__, ())
            if c.__name__ not in t:  # extend tuple
                _dict_classes[c.__module__] = t + (c.__name__,)
        else:  # not a dict-like class
            a = False
    return a  # all installed if True


_asizer = Asizer()


def asized(*objs, **opts):
    '''Return a tuple containing an **Asized** instance for each
       object passed as positional argument.

       The available options and defaults are:

            *above=0*      -- threshold for largest objects stats

            *align=8*      -- size alignment

            *code=False*   -- incl. (byte)code size

            *cutoff=10*    -- limit large objects or profiles stats

            *derive=False* -- derive from super type

            *detail=0*     -- Asized refs level

            *frames=False* -- ignore stack frame objects

            *ignored=True* -- ignore certain types

            *infer=False*  -- try to infer types

            *limit=100*    -- recursion limit

            *stats=0*      -- print statistics

       If only one object is given, the return value is the **Asized**
       instance for that object.  Otherwise, the length of the returned
       tuple matches the number of given objects.

       The **Asized** size of duplicate and ignored objects will be zero.

       Set *detail* to the desired referents level and *limit* to the
       maximum recursion depth.

       See function **asizeof** for descriptions of the other options.
    '''
    _asizer.reset(**opts)
    if objs:
        t = _asizer.asized(*objs)
        _asizer.print_stats(objs, opts=opts, sized=t)  # show opts as _kwdstr
        _asizer._clear()
    else:
        t = ()
    return t


def asizeof(*objs, **opts):
    '''Return the combined size (in bytes) of all objects passed
       as positional arguments.

       The available options and defaults are:

            *above=0*      -- threshold for largest objects stats

            *align=8*      -- size alignment

            *clip=80*      -- clip ``repr()`` strings

            *code=False*   -- incl. (byte)code size

            *cutoff=10*    -- limit large objects or profiles stats

            *derive=False* -- derive from super type

            *frames=False* -- ignore stack frame objects

            *ignored=True* -- ignore certain types

            *infer=False*  -- try to infer types

            *limit=100*    -- recursion limit

            *stats=0*      -- print statistics

       Set *align* to a power of 2 to align sizes.  Any value less
       than 2 avoids size alignment.

       If *all* is True and if no positional arguments are supplied.
       size all current gc objects, including module, global and stack
       frame objects.

       A positive *clip* value truncates all repr() strings to at
       most *clip* characters.

       The (byte)code size of callable objects like functions,
       methods, classes, etc. is included only if *code* is True.

       If *derive* is True, new types are handled like an existing
       (super) type provided there is one and only of those.

       By default certain base types like object, super, etc. are
       ignored.  Set *ignored* to False to include those.

       If *infer* is True, new types are inferred from attributes
       (only implemented for dict types on callable attributes
       as get, has_key, items, keys and values).

       Set *limit* to a positive value to accumulate the sizes of
       the referents of each object, recursively up to the limit.
       Using *limit=0* returns the sum of the flat sizes of the
       given objects.  High *limit* values may cause runtime errors
       and miss objects for sizing.

       A positive value for *stats* prints up to 9 statistics, (1)
       a summary of the number of objects sized and seen and a list
       of the largests objects with size over *above* bytes, (2) a
       simple profile of the sized objects by type and (3+) up to 6
       tables showing the static, dynamic, derived, ignored, inferred
       and dict types used, found respectively installed.
       The fractional part of the *stats* value (x 100) is the number
       of largest objects shown for (*stats*1.+) or the cutoff
       percentage for simple profiles for (*stats*=2.+).  For example,
       *stats=1.10* shows the summary and the 10 largest objects,
       also the default.

       See this module documentation for the definition of flat size.
    '''
    t, p, x = _objs_opts_x(objs, **opts)
    _asizer.reset(**p)
    if t:
        if x:  # don't size, profile or rank _getobjects tuple
            _asizer.exclude_objs(t)
        s = _asizer.asizeof(*t)
        _asizer.print_stats(objs=t, opts=opts)  # show opts as _kwdstr
        _asizer._clear()
    else:
        s = 0
    return s


def asizesof(*objs, **opts):
    '''Return a tuple containing the size (in bytes) of all objects
       passed as positional arguments.

       The available options and defaults are:

            *above=1024*   -- threshold for largest objects stats

            *align=8*      -- size alignment

            *clip=80*      -- clip ``repr()`` strings

            *code=False*   -- incl. (byte)code size

            *cutoff=10*    -- limit large objects or profiles stats

            *derive=False* -- derive from super type

            *frames=False* -- ignore stack frame objects

            *ignored=True* -- ignore certain types

            *infer=False*  -- try to infer types

            *limit=100*    -- recursion limit

            *stats=0*      -- print statistics

       See function **asizeof** for a description of the options.

       The length of the returned tuple equals the number of given
       objects.

       The size of duplicate and ignored objects will be zero.
    '''
    _asizer.reset(**opts)
    if objs:
        t = _asizer.asizesof(*objs)
        _asizer.print_stats(objs, opts=opts, sizes=t)  # show opts as _kwdstr
        _asizer._clear()
    else:
        t = ()
    return t


def _typedefof(obj, save=False, **opts):
    '''Get the typedef for an object.
    '''
    k = _objkey(obj)
    v = _typedefs.get(k, None)
    if not v:  # new typedef
        v = _typedef(obj, **opts)
        if save:
            _typedefs[k] = v
    return v


def basicsize(obj, **opts):
    '''Return the basic size of an object (in bytes).

       The available options and defaults are:

           *derive=False* -- derive type from super type

           *infer=False*  -- try to infer types

           *save=False*   -- save the object's type definition if new

       See this module documentation for the definition of *basic size*.
    '''
    b = t = _typedefof(obj, **opts)
    if t:
        b = t.base
    return b


def flatsize(obj, align=0, **opts):
    '''Return the flat size of an object (in bytes), optionally aligned
       to the given power of 2.

       See function **basicsize** for a description of other available options.

       See this module documentation for the definition of *flat size*.
    '''
    f = t = _typedefof(obj, **opts)
    if t:
        if align > 1:
            m = align - 1
            if (align & m) != 0:
                raise ValueError('invalid option: %s=%r' % ('align', align))
        else:
            m = 0
        f = t.flat(obj, mask=m)
    return f


def itemsize(obj, **opts):
    '''Return the item size of an object (in bytes).

       See function **basicsize** for a description of the available options.

       See this module documentation for the definition of *item size*.
    '''
    i = t = _typedefof(obj, **opts)
    if t:
        i, v = t.item, t.vari
        if v and i == _sizeof_Cbyte:
            i = getattr(obj, v, i)
    return i


def leng(obj, **opts):
    '''Return the length of an object (in items).

       See function **basicsize** for a description of the available options.
    '''
    n = t = _typedefof(obj, **opts)
    if t:
        n = t.leng
        if n and _callable(n):
            i, v, n = t.item, t.vari, n(obj)
            if v and i == _sizeof_Cbyte:
                i = getattr(obj, v, i)
                if i > _sizeof_Cbyte:
                    n = n // i
    return n


def named_refs(obj, **opts):
    '''Return all named **referents** of an object (re-using
       functionality from **asizeof**).

       Does not return un-named *referents*, e.g. objects in a list.

       See function **basicsize** for a description of the available options.
    '''
    rs = []
    v = _typedefof(obj, **opts)
    if v:
        v = v.refs
        if v and _callable(v):
            for r in v(obj, True):
                try:
                    rs.append((r.name, r.ref))
                except AttributeError:
                    pass
    return rs


def refs(obj, **opts):
    '''Return (a generator for) specific *referents* of an object.

       See function **basicsize** for a description of the available options.
    '''
    v = _typedefof(obj, **opts)
    if v:
        v = v.refs
        if v and _callable(v):
            v = v(obj, False)
    return v


if __name__ == '__main__':

    if '-v' in sys.argv:
        import platform
        print('%s %s (Python %s %s)' % (__file__, __version__,
                                        sys.version.split()[0],
                                        platform.architecture()[0]))

    elif '-types' in sys.argv:  # print static _typedefs
        n = len(_typedefs)
        w = len(str(n)) * ' '
        _printf('%s%d type definitions: %s and %s, kind ... %s', linesep,
                 n, 'basic-', 'itemsize (leng)', '-type[def]s')
        for k, td in sorted((_prepr(k), td) for k, td in _items(_typedefs)):
            desc = '%(base)s and %(item)s%(leng)s, %(kind)s%(code)s' % td.format()
            _printf('%s %s: %s', w, k, desc)

    else:
        import gc
        collect = False
        if '-gc' in sys.argv:
            collect = True
            gc.collect()

        frames = '-frames' in sys.argv

        # just an example
        asizeof(all=True, frames=frames, stats=1, above=1024)  # print summary + 10 largest

        if collect:
            print('gc.collect() %d' % (gc.collect(),))

# License from the initial version of this source file follows:

# --------------------------------------------------------------------
#       Copyright (c) 2002-2019 -- ProphICy Semiconductor, Inc.
#                        All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# - Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# - Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# - Neither the name of ProphICy Semiconductor, Inc. nor the names
#   of its contributors may be used to endorse or promote products
#   derived from this software without specific prior written
#   permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.
# --------------------------------------------------------------------
