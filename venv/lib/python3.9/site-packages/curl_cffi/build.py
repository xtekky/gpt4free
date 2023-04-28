import os
import platform

from cffi import FFI

ffibuilder = FFI()
# arch = "%s-%s" % (os.uname().sysname, os.uname().machine)
uname = platform.uname()


ffibuilder.set_source(
    "curl_cffi._wrapper",
    """
        #include "shim.h"
    """,
    libraries=["curl-impersonate-chrome"] if uname.system != "Windows" else ["libcurl"],
    library_dirs=[
        "/Users/runner/work/_temp/install/lib"
        if uname.system == "Darwin" and uname.machine == "x86_64"
        else "./lib" if uname.system == "Windows"
        else "/usr/local/lib"  # Linux and macOS arm64
    ],
    source_extension=".c",
    include_dirs=[
        os.path.join(os.path.dirname(__file__), "include"),
        os.path.join(os.path.dirname(__file__), "ffi"),
    ],
    sources=[
        os.path.join(os.path.dirname(__file__), "ffi/shim.c"),
    ],
    # extra_link_args=["-Wl,-rpath,$ORIGIN/../libcurl/" + arch],
)

with open(os.path.join(os.path.dirname(__file__), "ffi/cdef.c")) as f:
    cdef_content = f.read()
    ffibuilder.cdef(cdef_content)


if __name__ == "__main__":
    ffibuilder.compile(verbose=False)
