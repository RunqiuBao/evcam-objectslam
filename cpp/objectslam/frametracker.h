#ifndef EVENTOBJECTSLAM_FRAMETRACKER_H
#define EVENTOBJECTSLAM_FRAMETRACKER_H

#include "camera.h"

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


private:
    const camera::CameraBase* _camera; 

};


};  // end of namespace eventobjectslam

#endif  // EVENTOBJECTSLAM_FRAMETRACKER_H