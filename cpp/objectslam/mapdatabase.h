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

    void AddKeyFrame(std::shared_ptr<KeyFrame> pKeyFrame);

    void AddLandMark(std::shared_ptr<LandMark> pLandmark);

    // scan existing landmarks for ones that are within fov of input keyFrame.
    std::vector<std::shared_ptr<LandMark>> GetVisibleLandmarks(std::shared_ptr<KeyFrame> pRefKeyFrame);

    std::vector<std::shared_ptr<KeyFrame>> GetAllKeyframes() const;
    std::vector<std::shared_ptr<LandMark>> GetAllLandmarks() const;

    size_t _maxKeyFrameID = 0;
    //! IDs and keyframes
    std::unordered_map<unsigned int, std::shared_ptr<KeyFrame>> _keyframes;

    //! IDs and landmarks
    std::unordered_map<unsigned int, std::shared_ptr<LandMark>> _landmarks;

private:
    // mutex for mutal exclusion controll between class methods called in different threads
    mutable std::mutex _mtxMapAccess;

};

}  // end of namespace eventobjectslam

#endif