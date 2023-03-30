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
set(CMAKE_BUILD_TYPE "Debug")

# build 3rdparty
# list(APPEND PKG_CONFIG_PATH "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/log4cxx/install/lib/pkgconfig")
# list(APPEND PATH "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/log4cxx/install/lib/cmake")
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/log4cxx/install/" ${CMAKE_PREFIX_PATH})
find_package(log4cxx REQUIRED)
set(LOG4CXX_CONFIGURATION "${CMAKE_CURRENT_SOURCE_DIR}/log4cxx.properties")

# include Eigen
set(Eigen3_DIR ${CMAKE_CURRENT_SOURCE_DIR}/eigen/cmake/)
find_package (Eigen3 REQUIRED NO_MODULE)

# include pybind11
set(pybind11_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/pybind11/build/pybind11install/share/cmake/pybind11/")
message("pybind11 path: ${pybind11_DIR}")
find_package(pybind11 REQUIRED)
# include_directories("${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/pybind11/build/pybind11install/include/")
message(STATUS "Found pybind11 v${pybind11_VERSION}: ${pybind11_INCLUDE_DIRS}")

# include opencv
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(TDO_INSTALL_PATH "${CMAKE_CURRENT_BINARY_DIR}/tdoinstall")
set(CMAKE_INSTALL_PREFIX "${TDO_INSTALL_PATH}")

# add_subdirectory(3rdparty)
add_subdirectory(cpp)
add_subdirectory(examples)

# install python
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

