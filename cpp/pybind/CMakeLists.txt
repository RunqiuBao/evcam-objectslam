set(PYBIND11_CPP_STANDARD -std=c++17)
set(PYBIND11_PYTHON_VERSION 3.8)

find_package (Python 3.8 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

add_library(tdopybind SHARED pybindcommon.cpp)

target_include_directories(tdopybind PRIVATE ${Python_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR} ${pybind11_INCLUDE_DIRS})

target_link_libraries (tdopybind PRIVATE ${Python_LIBRARIES} Eigen3::Eigen pybind11::embed pybind11::module)

# install
install(TARGETS tdopybind
        RUNTIME DESTINATION lib)

set(TDOPYBIND_HEADER_FILES
    pybindcommon.h
)

install(FILES ${TDOPYBIND_HEADER_FILES} DESTINATION include)
