#ifndef EVENTOBJECTSLAM_MAPDATABASE_H
#define EVENTOBJECTSLAM_MAPDATABASE_H

#include "keyframe.h"
#include "landmark.h"

#include <mutex>
#include <unordered_map>

namespace eventobjectslam{


/*
 *  (1) hold all the keyframes
 *  (2) hold all the landmarks
 *  (3) publish keyframes and landmarks to pangolin viewer
 *
 */
class MapDataBase {

public:
    MapDataBase();

    // ~MapDataBase();

    void AddKeyFrame(KeyFrame* keyFrame);

    void AddLandMark(LandMark* landMark);

private:
    // mutex for mutal exclusion controll between class methods called in different threads
    mutable std::mutex _mtxMapAccess;

    //! IDs and keyframes
    std::unordered_map<unsigned int, KeyFrame*> _keyFrames;

    //! IDs and landmarks
    std::unordered_map<unsigned int, LandMark*> _landMarks;

}

}  // end of namespace eventobjectslam

#endif