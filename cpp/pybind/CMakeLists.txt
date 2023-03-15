add_library(tdopybind SHARED pybindcommon.cpp)

target_include_directories(tdopybind PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries (tdopybind Eigen3::Eigen pybind11::module)
