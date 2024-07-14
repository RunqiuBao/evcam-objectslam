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
    DESCRIPTION "An object base visual SLAM library for event camera"
    LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_BUILD_TYPE "Debug") // Note: control build type in the build script.
message(STATUS "set CMAKE_BUILD_TYPE to ${CMAKE_BUILD_TYPE}")

# thread library
find_package(Threads REQUIRED)

# build 3rdparty
set(CMAKE_PREFIX_PATH "${INSTALLED_LOG4CXX_LIB_PATH}/cmake/" ${CMAKE_PREFIX_PATH})
find_package(log4cxx REQUIRED)

# include Eigen
set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/eigen")
find_package (Eigen3 3.4 REQUIRED)

# include pybind11
set(pybind11_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/pybind11/pybind11install/share/cmake/pybind11/")
find_package(pybind11 REQUIRED)
# include_directories("${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/pybind11/build/pybind11install/include/")
# message(STATUS "Found pybind11 v${pybind11_VERSION}: ${pybind11_INCLUDE_DIRS}")

# include opencv
set(CMAKE_PREFIX_PATH "${INSTALLED_OPENCV4_LIB_PATH}/cmake/" ${CMAKE_PREFIX_PATH})
find_package( OpenCV 4.7.0 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# include rapidjson
set(CMAKE_PREFIX_PATH "${INSTALLED_RAPIDJSON_LIB_PATH}/cmake/" ${CMAKE_PREFIX_PATH})
find_package(RapidJSON)

# include boost
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

# include pcl
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})

# include pangolin
list(INSERT CMAKE_PREFIX_PATH 0 "${INSTALLED_PANGOLIN_LIB_PATH}/cmake/")  # Add your preferred directory to the beginning of CMAKE_PREFIX_PATH
find_package(Pangolin REQUIRED)
if (Pangolin_FOUND)
    set(Pangolin_INCLUDE_DIRS "${INSTALLED_PANGOLIN_LIB_PATH}/../include/")

    message(STATUS "Pangolin Found! - ${Pangolin_DIR}")
    message(STATUS "Pangolin Found! - ${Pangolin_INCLUDE_DIRS}")
    message(STATUS "Pangolin Found! - ${Pangolin_LIBRARIES}")
endif()

# include g2o
LIST( APPEND CMAKE_MODULE_PATH ${INSTALLED_G2O_LIB_PATH}/../../cmake_modules )
set( G2O_ROOT ${INSTALLED_G2O_LIB_PATH}/../ )
find_package(G2O REQUIRED)
if (G2O_FOUND)
    set(G2O_INCLUDE_DIRS "${INSTALLED_G2O_LIB_PATH}/../include/")
    set(G2O_LIBS
            ${G2O_CORE_LIBRARY}
            ${G2O_STUFF_LIBRARY}
            ${G2O_SOLVER_DENSE}
            ${G2O_SOLVER_EIGEN}
            ${G2O_SOLVER_CHOLMOD}
            ${G2O_SOLVER_CSPARSE}
            ${G2O_SOLVER_CSPARSE_EXTENSION}
            ${G2O_TYPES_SBA}
            ${G2O_TYPES_SIM3})
    message(STATUS "g2o Found! - ${INSTALLED_G2O_LIB_PATH}/../")
    message(STATUS "g2o Found! - G2O_LIBS - ${G2O_LIBS}")
else()
    message(FATAL_ERROR "g2o not Found!")
endif()


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

