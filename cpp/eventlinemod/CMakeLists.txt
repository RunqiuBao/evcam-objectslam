add_library(EventLinemod SHARED eventlinemod.cpp eventlinemodtemplatemanager.cpp)

target_include_directories(EventLinemod PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/3rdparty/rapidjson/build/rapidjsoninstall/include/)

message("baodebug: ${CMAKE_SOURCE_DIR}/cpp/common/logging.h")
target_link_libraries(EventLinemod PRIVATE log4cxx ${OpenCV_LIBS} Eigen3::Eigen)

install(TARGETS EventLinemod
        RUNTIME DESTINATION lib)

set(TDO_EVENTLINEMOD_HEADER_FILES
    eventlinemod.h
    eventlinemodtemplatemanager.h
    Box.h
    Quadtree.h
    Vector2.h
)

install(FILES ${TDO_EVENTLINEMOD_HEADER_FILES} DESTINATION include)
