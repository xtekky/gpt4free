"""
Apply Debian-specific patches to distutils commands.

Extracts the customized behavior from patches as reported
in pypa/distutils#2 and applies those customizations (except
for scheme definitions) to those commands.

Place this module somewhere in sys.path to take effect.
"""

import os
import sys
import sysconfig

import distutils.sysconfig
import distutils.command.install as orig_install
import distutils.command.install_egg_info as orig_install_egg_info
from distutils.command.install_egg_info import (
    to_filename,
    safe_name,
    safe_version,
    )
from distutils.errors import DistutilsOptionError


class install(orig_install.install):
    user_options = list(orig_install.install.user_options) + [
        ('install-layout=', None,
         "installation layout to choose (known values: deb, unix)"),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.prefix_option = None
        self.install_layout = None

    def select_scheme(self, name):
        if name == "posix_prefix":
            if self.install_layout:
                if self.install_layout.lower() in ['deb']:
                    name = "deb_system"
                elif self.install_layout.lower() in ['unix']:
                    name = "posix_prefix"
                else:
                    raise DistutilsOptionError(
                        "unknown value for --install-layout")
            elif ((self.prefix_option and
                   os.path.normpath(self.prefix) != '/usr/local')
                  or is_virtual_environment()):
                name = "posix_prefix"
            else:
                if os.path.normpath(self.prefix) == '/usr/local':
                    self.prefix = self.exec_prefix = '/usr'
                    self.install_base = self.install_platbase = '/usr'
                name = "posix_local"
        super().select_scheme(name)

    def finalize_unix(self):
        self.prefix_option = self.prefix
        super().finalize_unix()


class install_egg_info(orig_install_egg_info.install_egg_info):
    user_options = list(orig_install_egg_info.install_egg_info.user_options) + [
        ('install-layout', None, "custom installation layout"),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.prefix_option = None
        self.install_layout = None

    def finalize_options(self):
        self.set_undefined_options('install',('install_layout','install_layout'))
        self.set_undefined_options('install',('prefix_option','prefix_option'))
        super().finalize_options()

    @property
    def basename(self):
        if self.install_layout:
            if not self.install_layout.lower() in ['deb', 'unix']:
                raise DistutilsOptionError(
                    "unknown value for --install-layout")
            no_pyver = (self.install_layout.lower() == 'deb')
        elif self.prefix_option:
            no_pyver = False
        else:
            no_pyver = True
        if no_pyver:
            basename = "%s-%s.egg-info" % (
                to_filename(safe_name(self.distribution.get_name())),
                to_filename(safe_version(self.distribution.get_version()))
                )
        else:
            basename = "%s-%s-py%d.%d.egg-info" % (
                to_filename(safe_name(self.distribution.get_name())),
                to_filename(safe_version(self.distribution.get_version())),
                *sys.version_info[:2]
            )
        return basename


def is_virtual_environment():
    return sys.base_prefix != sys.prefix or hasattr(sys, "real_prefix")


def _posix_lib(standard_lib, libpython, early_prefix, prefix):
    is_default_prefix = not early_prefix or os.path.normpath(early_prefix) in ('/usr', '/usr/local')
    if standard_lib:
        return libpython
    elif is_default_prefix and not is_virtual_environment():
        return os.path.join(prefix, "lib", "python3", "dist-packages")
    else:
        return os.path.join(libpython, "site-packages")


def _inject_headers(name, scheme):
    """
    Given a scheme name and the resolved scheme,
    if the scheme does not include headers, resolve
    the fallback scheme for the name and use headers
    from it. pypa/distutils#88

    headers: module headers install location (posix_local is /local/ prefixed)
    include: cpython headers (Python.h)
    See also: bpo-44445
    """
    if 'headers' not in scheme:
        if name == 'posix_prefix':
            headers = scheme['include']
        else:
            headers = orig_install.INSTALL_SCHEMES['posix_prefix']['headers']
        if name == 'posix_local' and '/local/' not in headers:
            headers = headers.replace('/include/', '/local/include/')
        scheme['headers'] = headers
    return scheme


def load_schemes_wrapper(_load_schemes):
    """
    Implement the _inject_headers modification, above, but before
    _inject_headers() was introduced, upstream. So, slower and messier.
    """
    def wrapped_load_schemes():
        schemes = _load_schemes()
        for name, scheme in schemes.items():
            _inject_headers(name, scheme)
        return schemes
    return wrapped_load_schemes


def add_debian_schemes(schemes):
    """
    Ensure that the custom schemes we refer to above are present in schemes.
    """
    for name in ('posix_prefix', 'posix_local', 'deb_system'):
        if name not in schemes:
            scheme = sysconfig.get_paths(name, expand=False)
            schemes[name] = _inject_headers(name, scheme)


def apply_customizations():
    orig_install.install = install
    orig_install_egg_info.install_egg_info = install_egg_info
    distutils.sysconfig._posix_lib = _posix_lib

    if hasattr(orig_install, '_inject_headers'):
        # setuptools-bundled distutils >= 60.0.5
        orig_install._inject_headers = _inject_headers
    elif hasattr(orig_install, '_load_schemes'):
        # setuptools-bundled distutils >= 59.2.0
        orig_install._load_schemes = load_schemes_wrapper(orig_install._load_schemes)
    else:
        # older version with only statically defined schemes
        # this includes the version bundled with Python 3.10 that has our
        # schemes already included
        add_debian_schemes(orig_install.INSTALL_SCHEMES)


apply_customizations()
