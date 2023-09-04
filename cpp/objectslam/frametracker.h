#ifndef EVENTOBJECTSLAM_FRAMETRACKER_H
#define EVENTOBJECTSLAM_FRAMETRACKER_H

#include "objectslam.h"
#include "camera.h"
#include "keyframe.h"
#include "frame.h"
#include "mapdatabase.h"
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
    FrameTracker(std::shared_ptr<camera::CameraBase> camera)
    : _camera(camera)
    {}

    bool DoMotionBasedTrack(Frame& currentFrame, const Frame& lastFrame, Mat44_t& velocity) const;

    bool Do2DTrackingBasedTrack(Frame& currentFrame, const Frame& lastFrame, Mat44_t& velocity) const;

    void CreateNewLandmarks(std::shared_ptr<KeyFrame> pRefKeyFrame, std::shared_ptr<MapDataBase> mapDb, const std::shared_ptr<object::ObjectBase> pObjectInfo);

    std::shared_ptr<KeyFrame> _pRefKeyframe;

    std::string _sStereoSequencePathForDebug;

private:
    std::shared_ptr<camera::CameraBase> _camera;


};


};  // end of namespace eventobjectslam

#endif  // EVENTOBJECTSLAM_FRAMETRACKER_H