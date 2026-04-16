cmake_minimum_required(VERSION 3.10)
project(zkGPT)
set(CMAKE_CXX_STANDARD 14)

# Default optimization level to O0 if not specified
if(NOT DEFINED OPT_LEVEL)
    set(OPT_LEVEL O0)
endif()

set(CMAKE_CXX_FLAGS "-mcmodel=large -${OPT_LEVEL} -lpthread -pthread")
link_directories(3rd/mcl)

include_directories(src)
include_directories(3rd)
include_directories(3rd/mcl/include)

add_subdirectory(src)
add_subdirectory(3rd/mcl)