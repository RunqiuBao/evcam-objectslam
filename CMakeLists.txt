cmake_minimum_required(VERSION 3.20)

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

add_subdirectory(cpp)

