#ifndef EVENTOBJECTSLAM_FRAMETRACKER_H
#define EVENTOBJECTSLAM_FRAMETRACKER_H

#include "objectslam.h"
#include "camera.h"
#include "frame.h"

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
    FrameTracker(camera::CameraBase* camera);

    bool DoBruteForceMatchBasedTrack(frame& currentFrame, const frame& lastFrame, const Mat44_t& velocity);

    bool DoMotionBasedTrack(frame& currentFrame, const frame& lastFrame, const Mat44_t& velocity);

private:
    const camera::CameraBase* _camera;

};


};  // end of namespace eventobjectslam

#endif  // EVENTOBJECTSLAM_FRAMETRACKER_H