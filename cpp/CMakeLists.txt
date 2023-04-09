include_directories($<TARGET_PROPERTY:log4cxx,INTERFACE_INCLUDE_DIRECTORIES> "${CMAKE_SOURCE_DIR}/cpp/common/")  # add log4cxx header files

add_subdirectory(common)
add_subdirectory(pybind)
add_subdirectory(eventlinemod)
