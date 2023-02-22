add_executable(myapp DetectObjectsByEventLinemod.cpp)

target_include_directories(myapp PRIVATE $<TARGET_PROPERTY:log4cxx,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(myapp PRIVATE log4cxx)

install(TARGETS myapp RUNTIME DESTINATION bin)
