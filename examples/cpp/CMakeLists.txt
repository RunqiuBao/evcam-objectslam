# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../cpp/common/ ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp/pybind/)

message("${CMAKE_CURRENT_SOURCE_DIR}/../../cpp/common")
add_executable(detectorRunner DetectObjectsByEventLinemod.cpp)

find_package (Python REQUIRED COMPONENTS Interpreter Development)

message("Python3_INCLUDE_DIRS ${Python_INCLUDE_DIRS}")
message("Python2_INCLUDE_DIRS ${Python2_INCLUDE_DIRS}")
target_include_directories(detectorRunner PRIVATE $<TARGET_PROPERTY:log4cxx,INTERFACE_INCLUDE_DIRECTORIES> ${Python_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp/pybind/)

get_target_property(tdocommon_type tdocommon TYPE)
get_target_property(tdopybind_type tdopybind TYPE)
if(NOT tdocommon_type)
    message("tdocommon not found: ${tdocommon}")
endif()
if(NOT tdopybind_type)
    message("tdopybind not found: ${tdopybind}")
endif()

# add_library( libtdopybind SHARED IMPORTED )  # same error as linking tdopybind
# set_target_properties( libtdopybind PROPERTIES IMPORTED_LOCATION ${TDO_INSTALL_PATH}/lib/libtdopybind.so )
target_link_libraries(detectorRunner PRIVATE tdocommon ${Python_LIBRARIES} Eigen3::Eigen pybind11::module tdopybind)

install(TARGETS detectorRunner 
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib)
