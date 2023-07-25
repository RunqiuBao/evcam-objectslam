#ifndef EVENTOBJECTSLAM_FRAMETRACKER_H
#define EVENTOBJECTSLAM_FRAMETRACKER_H

#include "objectslam.h"
#include "camera.h"
#include "frame.h"
#include <opencv2/opencv.hpp>

namespace eventobjectslam{

// tracker state
enum class TrackerStatus {
    NotInitialized,
    Initializing,
    Tracking,
    Lost
};


class FrameTracker {

public:
    // constructor
    FrameTracker(std::shared_ptr<camera::CameraBase> camera);

    Mat44_t DoBruteForceMatchBasedTrack(Frame& currentFrame, const Frame& lastFrame, const Mat44_t& velocity);

    Mat44_t DoMotionBasedTrack(Frame& currentFrame, const Frame& lastFrame, const Mat44_t& velocity);

private:
    std::shared_ptr<camera::CameraBase> _camera;

};


};  // end of namespace eventobjectslam

#endif  // EVENTOBJECTSLAM_FRAMETRACKER_H