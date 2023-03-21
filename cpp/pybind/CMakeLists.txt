find_package(pybind11 REQUIRED)
find_package (Python REQUIRED COMPONENTS Interpreter Development)

add_library(tdopybind SHARED pybindcommon.cpp)

message("pybind CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(tdopybind PRIVATE ${Python_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR} ${pybind11_INCLUDE_DIRS})
message("EIGEN_INCLUDE_DIRS: ${EIGEN_INCLUDE_DIRS}")
message("pybind11_INCLUDE_DIRS: ${pybind11_INCLUDE_DIRS}")

target_link_libraries (tdopybind PRIVATE ${Python_LIBRARIES} Eigen3::Eigen pybind11::embed pybind11::module)

# install
install(TARGETS tdopybind
        RUNTIME DESTINATION lib)

set(TDOPYBIND_HEADER_FILES
    pybindcommon.h
)

install(FILES ${TDOPYBIND_HEADER_FILES} DESTINATION include)
    