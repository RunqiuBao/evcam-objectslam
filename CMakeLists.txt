cmake_minimum_required(VERSION 3.16)

if(UNIX AND NOT APPLE)
    execute_process(COMMAND uname -p
        OUTPUT_VARIABLE PROCESSOR_ARCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(PROCESSOR_ARCH STREQUAL "aarch64")
        set(LINUX_AARCH64 TRUE)
    endif()
endif()

set(MYLIB_VERSION 0.0.1)
project(tool_detectobjects
    VERSION ${MYLIB_VERSION}
    DESCRIPTION "An object detection library for event camera"
    LANGUAGES C CXX
)
message(STATUS "tool_detectobjects ${MYLIB_VERSION}")

set(CMAKE_CXX_STANDARD 17)

# build 3rdparty
# list(APPEND PKG_CONFIG_PATH "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/log4cxx/install/lib/pkgconfig")
# list(APPEND PATH "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/log4cxx/install/lib/cmake")
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/log4cxx/install/" ${CMAKE_PREFIX_PATH})
find_package(log4cxx REQUIRED)
set(LOG4CXX_CONFIGURATION "${CMAKE_CURRENT_SOURCE_DIR}/log4cxx.properties")

# include pybind11
message("pybind11 include path: ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/pybind11/include/")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/pybind11/include/")
message("include pybind11 success!")

add_subdirectory(cpp)
add_subdirectory(examples)

# install
message("install start!")
execute_process(
    COMMAND bash -c "yes Y | python3 -m pip uninstall tool_eventdetectobjects"
    OUTPUT_VARIABLE outVar
)
message("uninstall tool_eventdetectobjects python package:\n${outVar}")
execute_process(
    COMMAND bash -c "cd ${CMAKE_CURRENT_SOURCE_DIR}/python && python3 -m build && cd dist/ && python3 -m pip install tool_eventdetectobjects-0.0.1-py3-none-any.whl"
    OUTPUT_VARIABLE outVar
)
message("install tool_eventdetectobjects python package:\n${outVar}")

