include_directories($<TARGET_PROPERTY:log4cxx,INTERFACE_INCLUDE_DIRECTORIES> "${CMAKE_SOURCE_DIR}/common/logging.h")  # add log4cxx header files

add_subdirectory(common)
add_subdirectory(eventlinemod)
