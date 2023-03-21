add_library(EventLinemod SHARED EventLinemod.cpp)

target_include_directories(EventLinemod INTERFACE ${CMAKE_CURRENT_SOURCE_DIR} 
                                        PRIVATE $<TARGET_PROPERTY:log4cxx,INTERFACE_INCLUDE_DIRECTORIES>)

target_link_libraries(EventLinemod log4cxx)

install(TARGETS EventLinemod
        RUNTIME DESTINATION lib)

set(TDO_EVENTLINEMOD_HEADER_FILES
    EventLinemod.h
)
    
install(FILES ${TDO_EVENTLINEMOD_HEADER_FILES} DESTINATION include)
