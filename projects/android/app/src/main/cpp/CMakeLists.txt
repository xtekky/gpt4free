cmake_minimum_required(VERSION 3.22.1)

project(g4f_container_native)

# AGP's toolchain adds -latomic -lm to C++ shared libraries. The linkfix
# preload must have no DT_NEEDED entries so it can be loaded by both bionic
# and glibc processes; all of its libc symbols are resolved by its host.
set(CMAKE_CXX_STANDARD_LIBRARIES "")

add_library(g4f_pty SHARED pty.cpp)

find_library(log-lib log)
target_link_libraries(g4f_pty ${log-lib})

# PRoot cannot create hard links in app data on some Android kernels, which
# breaks dpkg's status-old backup. This preload is loaded into every container
# process and turns link() into a regular copy. It only uses libc symbols that
# the loading process already provides, so it must not link against libc++.
add_library(g4f_linkfix SHARED linkfix.cpp)
set_target_properties(g4f_linkfix PROPERTIES CXX_STANDARD 17)
target_compile_options(g4f_linkfix PRIVATE -fno-stack-protector)
target_link_options(g4f_linkfix PRIVATE
    -nostdlib
    -Wl,--unresolved-symbols=ignore-all)
