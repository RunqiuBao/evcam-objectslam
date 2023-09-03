#ifndef EVENTOBJECTSLAM_LANDMARK_H
#define EVENTOBJECTSLAM_LANDMARK_H

#include "objectslam.h"
#include "keyframe.h"

#include <mutex>

namespace eventobjectslam {

class KeyFrame;  // Note: due to mutual reference.

class LandMark {

public:
    LandMark(const Mat44_t poseCurrentFrameInWorld, const std::shared_ptr<object::ObjectBase> pObjectInfo);

    unsigned int _numObservations = 0;

    Mat44_t _poseLandmarkInWorld;

    unsigned int _landmarkID;

    const std::shared_ptr<object::ObjectBase> _pObjectInfo;
    Eigen::MatrixXf _vertices3DInLandmark;  // 3D bounding box in landmark frame.

private:
    std::unordered_map<std::shared_ptr<KeyFrame>, unsigned int> _observations;

    mutable std::mutex _mtxObservations;

    static std::atomic<unsigned int> _nextID;
};

}  // end of namespace eventobjectslam

#endif  // EVENTOBJECTSLAM_LANDMARK_H
