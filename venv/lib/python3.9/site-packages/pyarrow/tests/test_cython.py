# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import shutil
import subprocess
import sys

import pytest

import pyarrow as pa
import pyarrow.tests.util as test_util


here = os.path.dirname(os.path.abspath(__file__))
test_ld_path = os.environ.get('PYARROW_TEST_LD_PATH', '')
if os.name == 'posix':
    compiler_opts = ['-std=c++17']
elif os.name == 'nt':
    compiler_opts = ['-D_ENABLE_EXTENDED_ALIGNED_STORAGE', '/std:c++17']
else:
    compiler_opts = []


setup_template = """if 1:
    from setuptools import setup
    from Cython.Build import cythonize

    import numpy as np

    import pyarrow as pa

    ext_modules = cythonize({pyx_file!r})
    compiler_opts = {compiler_opts!r}
    custom_ld_path = {test_ld_path!r}

    for ext in ext_modules:
        # XXX required for numpy/numpyconfig.h,
        # included from arrow/python/api.h
        ext.include_dirs.append(np.get_include())
        ext.include_dirs.append(pa.get_include())
        ext.libraries.extend(pa.get_libraries())
        ext.library_dirs.extend(pa.get_library_dirs())
        if custom_ld_path:
            ext.library_dirs.append(custom_ld_path)
        ext.extra_compile_args.extend(compiler_opts)
        print("Extension module:",
              ext, ext.include_dirs, ext.libraries, ext.library_dirs)

    setup(
        ext_modules=ext_modules,
    )
"""


def check_cython_example_module(mod):
    arr = pa.array([1, 2, 3])
    assert mod.get_array_length(arr) == 3
    with pytest.raises(TypeError, match="not an array"):
        mod.get_array_length(None)

    scal = pa.scalar(123)
    cast_scal = mod.cast_scalar(scal, pa.utf8())
    assert cast_scal == pa.scalar("123")
    with pytest.raises(NotImplementedError,
                       match="casting scalars of type int64 to type list"):
        mod.cast_scalar(scal, pa.list_(pa.int64()))


@pytest.mark.cython
def test_cython_api(tmpdir):
    """
    Basic test for the Cython API.
    """
    # Fail early if cython is not found
    import cython  # noqa

    with tmpdir.as_cwd():
        # Set up temporary workspace
        pyx_file = 'pyarrow_cython_example.pyx'
        shutil.copyfile(os.path.join(here, pyx_file),
                        os.path.join(str(tmpdir), pyx_file))
        # Create setup.py file
        setup_code = setup_template.format(pyx_file=pyx_file,
                                           compiler_opts=compiler_opts,
                                           test_ld_path=test_ld_path)
        with open('setup.py', 'w') as f:
            f.write(setup_code)

        # ARROW-2263: Make environment with this pyarrow/ package first on the
        # PYTHONPATH, for local dev environments
        subprocess_env = test_util.get_modified_env_with_pythonpath()

        # Compile extension module
        subprocess.check_call([sys.executable, 'setup.py',
                               'build_ext', '--inplace'],
                              env=subprocess_env)

        # Check basic functionality
        orig_path = sys.path[:]
        sys.path.insert(0, str(tmpdir))
        try:
            mod = __import__('pyarrow_cython_example')
            check_cython_example_module(mod)
        finally:
            sys.path = orig_path

        # Check the extension module is loadable from a subprocess without
        # pyarrow imported first.
        code = """if 1:
            import sys
            import os

            try:
                # Add dll directory was added on python 3.8
                # and is required in order to find extra DLLs
                # only for win32
                for dir in {library_dirs}:
                    os.add_dll_directory(dir)
            except AttributeError:
                pass

            mod = __import__({mod_name!r})
            arr = mod.make_null_array(5)
            assert mod.get_array_length(arr) == 5
            assert arr.null_count == 5
        """.format(mod_name='pyarrow_cython_example',
                   library_dirs=pa.get_library_dirs())

        path_var = None
        if sys.platform == 'win32':
            if not hasattr(os, 'add_dll_directory'):
                # Python 3.8 onwards don't check extension module DLLs on path
                # we have to use os.add_dll_directory instead.
                delim, path_var = ';', 'PATH'
        elif sys.platform == 'darwin':
            delim, path_var = ':', 'DYLD_LIBRARY_PATH'
        else:
            delim, path_var = ':', 'LD_LIBRARY_PATH'

        if path_var:
            paths = sys.path
            paths += pa.get_library_dirs()
            paths += [subprocess_env.get(path_var, '')]
            paths = [path for path in paths if path]
            subprocess_env[path_var] = delim.join(paths)
        subprocess.check_call([sys.executable, '-c', code],
                              stdout=subprocess.PIPE,
                              env=subprocess_env)


@pytest.mark.cython
def test_visit_strings(tmpdir):
    with tmpdir.as_cwd():
        # Set up temporary workspace
        pyx_file = 'bound_function_visit_strings.pyx'
        shutil.copyfile(os.path.join(here, pyx_file),
                        os.path.join(str(tmpdir), pyx_file))
        # Create setup.py file
        setup_code = setup_template.format(pyx_file=pyx_file,
                                           compiler_opts=compiler_opts,
                                           test_ld_path=test_ld_path)
        with open('setup.py', 'w') as f:
            f.write(setup_code)

        subprocess_env = test_util.get_modified_env_with_pythonpath()

        # Compile extension module
        subprocess.check_call([sys.executable, 'setup.py',
                               'build_ext', '--inplace'],
                              env=subprocess_env)

    sys.path.insert(0, str(tmpdir))
    mod = __import__('bound_function_visit_strings')

    strings = ['a', 'b', 'c']
    visited = []
    mod._visit_strings(strings, visited.append)

    assert visited == strings

    with pytest.raises(ValueError, match="wtf"):
        def raise_on_b(s):
            if s == 'b':
                raise ValueError('wtf')

        mod._visit_strings(strings, raise_on_b)
