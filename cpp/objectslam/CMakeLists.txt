add_library(ObjectSlam SHARED frametracker.cpp system.cpp object.cpp camera.cpp mathutils.cpp frame.cpp keyframe.cpp)

target_include_directories(ObjectSlam PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/3rdparty/rapidjson/build/rapidjsoninstall/include/)
target_link_libraries(ObjectSlam PRIVATE log4cxx ${OpenCV_LIBS} Eigen3::Eigen ${Boost_LIBRARIES})

install(TARGETS ObjectSlam
        RUNTIME DESTINATION lib)

set(TDO_OBJECTSLAM_HEADER_FILES
    camera.h
    frame.h
    frametracker.h
    objectslam.h
    system.h
)

install(FILES ${TDO_OBJECTSLAM_HEADER_FILES} DESTINATION include)
