#ifndef EVENTOBJECTSLAM_LANDMARK_H
#define EVENTOBJECTSLAM_LANDMARK_H

#include "objectslam.h"
#include "keyframe.h"

#include <mutex>

namespace eventobjectslam {

class KeyFrame;  // Note: due to mutual reference.

class LandMark {

public:
    LandMark(const Mat44_t pose_wc);

    unsigned int _numObservations = 0;

    Mat44_t _pose_wc;

    unsigned int _landmarkID;

private:
    std::unordered_map<std::shared_ptr<KeyFrame>, unsigned int> _observations;

    mutable std::mutex _mtxObservations;

    static std::atomic<unsigned int> _nextID;
};

}  // end of namespace eventobjectslam

#endif  // EVENTOBJECTSLAM_LANDMARK_H
