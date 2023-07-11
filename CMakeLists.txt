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

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Debug")

# build 3rdparty
set(CMAKE_PREFIX_PATH "${INSTALLED_LOG4CXX_LIB_PATH}/cmake/" ${CMAKE_PREFIX_PATH})
find_package(log4cxx REQUIRED)

# include Eigen
set(Eigen3_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/eigen/cmake/)
find_package (Eigen3 REQUIRED NO_MODULE)

# include pybind11
set(pybind11_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/pybind11/build/pybind11install/share/cmake/pybind11/")
find_package(pybind11 REQUIRED)
# include_directories("${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/pybind11/build/pybind11install/include/")
message(STATUS "Found pybind11 v${pybind11_VERSION}: ${pybind11_INCLUDE_DIRS}")

# include opencv
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# include rapidjson
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/3rdparty/rapidjson/build/rapidjsoninstall/")
find_package(RapidJSON)

set(TDO_INSTALL_PATH "${INSTALL_PATH_ALLTARGETS}")
set(CMAKE_INSTALL_PREFIX "${TDO_INSTALL_PATH}")

# add_subdirectory(3rdparty)
add_subdirectory(cpp)
add_subdirectory(examples)

# # install python
# message("install start!")
# execute_process(
#     COMMAND bash -c "yes Y | python3 -m pip uninstall tool_eventdetectobjects"
#     OUTPUT_VARIABLE outVar
# )
# message("uninstall tool_eventdetectobjects python package:\n${outVar}")
# execute_process(
#     COMMAND bash -c "cd ${CMAKE_CURRENT_SOURCE_DIR}/python && python3 -m build && cd dist/ && python3 -m pip install tool_eventdetectobjects-0.0.1-py3-none-any.whl"
#     OUTPUT_VARIABLE outVar
# )
# message("install tool_eventdetectobjects python package:\n${outVar}")

