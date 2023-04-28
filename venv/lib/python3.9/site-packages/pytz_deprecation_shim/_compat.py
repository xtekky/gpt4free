import sys

if sys.version_info[0] == 2:
    from . import _compat_py2 as _compat_impl
else:
    from . import _compat_py3 as _compat_impl

UTC = _compat_impl.UTC
get_timezone = _compat_impl.get_timezone
get_timezone_file = _compat_impl.get_timezone_file
get_fixed_offset_zone = _compat_impl.get_fixed_offset_zone
is_ambiguous = _compat_impl.is_ambiguous
is_imaginary = _compat_impl.is_imaginary
enfold = _compat_impl.enfold
get_fold = _compat_impl.get_fold
