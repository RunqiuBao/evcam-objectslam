add_library(tdocommon SHARED logging.cpp argparser.cpp)

message("tdocommon: CMAKE_CURRENT_SOURCE_DIR, ${CMAKE_CURRENT_SOURCE_DIR}.")
target_include_directories(tdocommon INTERFACE ${CMAKE_CURRENT_SOURCE_DIR} 
                                        PUBLIC $<TARGET_PROPERTY:log4cxx,INTERFACE_INCLUDE_DIRECTORIES>)

target_link_libraries(tdocommon PUBLIC log4cxx)

install(TARGETS tdocommon
        RUNTIME DESTINATION lib)

set(TDOCOMMON_HEADER_FILES
    argparser.h
    logging.h
)

install(FILES ${TDOCOMMON_HEADER_FILES} DESTINATION include)