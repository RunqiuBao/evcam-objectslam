add_library(tdocommon SHARED logging.cpp argparser.cpp)

message("tdocommon: CMAKE_CURRENT_SOURCE_DIR, ${CMAKE_CURRENT_SOURCE_DIR}.")
target_include_directories(tdocommon INTERFACE ${CMAKE_CURRENT_SOURCE_DIR} 
                                        PRIVATE $<TARGET_PROPERTY:log4cxx,INTERFACE_INCLUDE_DIRECTORIES>)

target_link_libraries(tdocommon log4cxx)