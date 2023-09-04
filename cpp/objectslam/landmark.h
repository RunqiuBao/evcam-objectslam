#ifndef EVENTOBJECTSLAM_LANDMARK_H
#define EVENTOBJECTSLAM_LANDMARK_H

#include "objectslam.h"
#include "keyframe.h"

#include <mutex>

namespace eventobjectslam {

class KeyFrame;  // Note: due to mutual reference.

class LandMark {

public:
    LandMark(const Mat44_t poseLandmarkInWorld, const std::shared_ptr<object::ObjectBase> pObjectInfo);

    unsigned int _numObservations = 0;

    Mat44_t _poseLandmarkInWorld;

    unsigned int _landmarkID;
    float _bestDetectionScore;  // if detection score becomes better, update detection orientation. 

    const std::shared_ptr<object::ObjectBase> _pObjectInfo;
    Eigen::MatrixXf _vertices3DInLandmark;  // 3D bounding box in landmark frame.

    void AddObservation(std::shared_ptr<KeyFrame> pRefKeyFrame, unsigned int idx);

private:
    std::unordered_map<std::shared_ptr<KeyFrame>, unsigned int> _observations_indices;  //Note: uint is the refOject index in this keyframe.

    mutable std::mutex _mtxObservations;

    static std::atomic<unsigned int> _nextID;
};

}  // end of namespace eventobjectslam

#endif  // EVENTOBJECTSLAM_LANDMARK_H
