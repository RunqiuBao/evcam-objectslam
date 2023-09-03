#ifndef EVENTOBJECTSLAM_KEYFRAME_H
#define EVENTOBJECTSLAM_KEYFRAME_H

#include "frame.h"
#include "object.h"
#include "landmark.h"

#include <opencv2/opencv.hpp>

namespace eventobjectslam{

class Frame;  // Note: due to mutual reference.
class LandMark;  // Note: due to mutual reference.

class KeyFrame {

public:
    KeyFrame(const std::shared_ptr<Frame> pRefFrame, const Mat44_t& refKeyFrameInWorldTransform);

    std::vector<std::shared_ptr<RefObject>> _refObjects;

    Mat44_t _pose_wc;  // Note: pose current to world

    std::shared_ptr<Frame> _pRefFrame;

    unsigned int _keyFrameID;
    static std::atomic<unsigned int> _nextID;

    // observed landmarks
    std::vector<std::shared_ptr<LandMark>> _observedLandmarks;

private:
    mutable std::mutex _mtxLandmarks;

};


}  // end of namespace eventobjectslam

#endif  // EVENTOBJECTSLAM_KEYFRAME_H