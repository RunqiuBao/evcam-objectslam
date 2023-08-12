#ifndef EVENTOBJECTSLAM_KEYFRAME_H
#define EVENTOBJECTSLAM_KEYFRAME_H

#include "frame.h"
#include "object.h"

#include <opencv2/opencv.hpp>

namespace eventobjectslam{

class Frame;  // Note: due to mutual reference.

class KeyFrame {

public:
    KeyFrame(const Frame& refFrame);

    std::vector<std::shared_ptr<LandMark>> landmarks;

    Mat44_t _pose_wc;  // Note: pose current to world

};


}  // end of namespace eventobjectslam

#endif  // EVENTOBJECTSLAM_KEYFRAME_H