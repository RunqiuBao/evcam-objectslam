include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../cpp/common)
message("${CMAKE_CURRENT_SOURCE_DIR}/../../cpp/common")
add_executable(myapp DetectObjectsByEventLinemod.cpp)

target_include_directories(myapp PRIVATE $<TARGET_PROPERTY:log4cxx,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(myapp PUBLIC tdocommon
                            PRIVATE log4cxx)

install(TARGETS myapp RUNTIME DESTINATION bin)
