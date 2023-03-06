include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../cpp/common)
message("${CMAKE_CURRENT_SOURCE_DIR}/../../cpp/common")
add_executable(detectorRunner DetectObjectsByEventLinemod.cpp)

find_package (Python REQUIRED COMPONENTS Interpreter Development)

message("Python3_INCLUDE_DIRS ${Python_INCLUDE_DIRS}")
message("Python2_INCLUDE_DIRS ${Python2_INCLUDE_DIRS}")
target_include_directories(detectorRunner PRIVATE $<TARGET_PROPERTY:log4cxx,INTERFACE_INCLUDE_DIRECTORIES> ${Python_INCLUDE_DIRS})
target_link_libraries(detectorRunner PUBLIC tdocommon
                            PRIVATE log4cxx ${Python_LIBRARIES})

install(TARGETS detectorRunner RUNTIME DESTINATION bin)
