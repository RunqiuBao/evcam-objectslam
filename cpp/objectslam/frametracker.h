#include "camera.h"

namespace eventobjectslam{

// tracker state
enum class Tracker_Status {
    NotInitialized,
    Initializing,
    Tracking,
    Lost
};


class FrameTracker {

public:
    // constructor
    FrameTracker(camera::base* camera);


private:
    const camera::base* _camera;

};


};  // end of namespace eventobjectslam