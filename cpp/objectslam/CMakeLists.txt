add_library(ObjectSlam SHARED 
        frametracker.cpp
        system.cpp
        object.cpp
        camera.cpp
        mathutils.cpp
        frame.cpp
        keyframe.cpp
        mapdatabase.cpp
        landmark.cpp
        graphnode.cpp
        semanticmapper.cpp
        )

add_subdirectory(pangolinviewer)

target_include_directories(ObjectSlam PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${INSTALLED_RAPIDJSON_LIB_PATH}/../include/
        ${Pangolin_INCLUDE_DIRS})

target_link_libraries(ObjectSlam PRIVATE
        Threads::Threads
        log4cxx
        ${OpenCV_LIBS}
        Eigen3::Eigen
        ${Boost_LIBRARIES})
target_link_libraries(ObjectSlam PUBLIC ${Pangolin_LIBRARIES})

install(TARGETS ObjectSlam
        RUNTIME DESTINATION lib)

# set(TDO_OBJECTSLAM_HEADER_FILES
#     camera.h
#     frame.h
#     frametracker.h
#     objectslam.h
#     system.h
#     pangolinviewer/viewer.h
# )

# install(FILES ${TDO_OBJECTSLAM_HEADER_FILES} DESTINATION include)
