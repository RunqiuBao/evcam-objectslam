add_executable(detectorRunner DetectObjectsByEventLinemod.cpp)
add_executable(slamRunner RunObjectSlam.cpp)

set(PYBIND11_CPP_STANDARD -std=c++17)
set(PYBIND11_PYTHON_VERSION 3.8)

find_package (Python 3.8 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

target_include_directories(detectorRunner PRIVATE $<TARGET_PROPERTY:log4cxx,INTERFACE_INCLUDE_DIRECTORIES> ${Python_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/cpp/pybind/ ${CMAKE_SOURCE_DIR}/cpp/eventlinemod/ ${CMAKE_SOURCE_DIR}/cpp/common/)
target_include_directories(slamRunner PRIVATE $<TARGET_PROPERTY:log4cxx,INTERFACE_INCLUDE_DIRECTORIES> ${CMAKE_SOURCE_DIR}/cpp/objectslam/ ${CMAKE_SOURCE_DIR}/cpp/common/)

get_target_property(tdocommon_type tdocommon TYPE)
get_target_property(tdopybind_type tdopybind TYPE)
if(NOT tdocommon_type)
    message("tdocommon not found: ${tdocommon}")
endif()
if(NOT tdopybind_type)
    message("tdopybind not found: ${tdopybind}")
endif()

target_link_libraries(detectorRunner PRIVATE EventLinemod tdocommon ${Python_LIBRARIES} Eigen3::Eigen pybind11::module tdopybind ${OpenCV_LIBS})
target_link_libraries(slamRunner PRIVATE ObjectSlam tdocommon Eigen3::Eigen ${OpenCV_LIBS})

install(TARGETS detectorRunner 
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib)

install(TARGETS slamRunner 
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib)
