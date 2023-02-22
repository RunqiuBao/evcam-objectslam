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

add_subdirectory(cpp)
add_subdirectory(examples)

