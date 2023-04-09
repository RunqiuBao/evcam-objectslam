add_library(EventLinemod SHARED eventlinemod.cpp eventlinemodtemplatemanager.cpp)

target_include_directories(EventLinemod PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

message("baodebug: ${CMAKE_SOURCE_DIR}/cpp/common/logging.h")
target_link_libraries(EventLinemod PRIVATE log4cxx ${OpenCV_LIBS} Eigen3::Eigen)

install(TARGETS EventLinemod
        RUNTIME DESTINATION lib)

set(TDO_EVENTLINEMOD_HEADER_FILES
    EventLinemod.h
)

install(FILES ${TDO_EVENTLINEMOD_HEADER_FILES} DESTINATION include)
