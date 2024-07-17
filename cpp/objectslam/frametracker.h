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

    bool DoMotionBasedTrack(Frame& currentFrame, const Frame& lastFrame, Mat44_t& velocity, const float minIoUToReject, const bool isDebug = false) const;

    bool Do2DTrackingBasedTrack(Frame& currentFrame, const Frame& lastFrame, Mat44_t& velocity, const bool isDebug = false) const;

    bool DoRelocalizeFromMap(Frame& currentFrame, const Frame& lastFrame, std::shared_ptr<MapDataBase> pMapDb, Mat44_t& velocity, const float minIoUToReject, const bool isDebug);

    void CreateNewLandmarks(std::shared_ptr<KeyFrame> pRefKeyFrame, std::shared_ptr<MapDataBase> mapDb, const bool isDebug = false);

    void SetTrackerStatus(const bool isInitialized){ _isInitialized = isInitialized; }

    bool GetTrackerStatus() { return _isInitialized; }

    std::shared_ptr<KeyFrame> _pRefKeyframe;

    std::string _sStereoSequencePathForDebug;

private:
    std::shared_ptr<camera::CameraBase> _camera;

    bool _isInitialized = false;


};


};  // end of namespace eventobjectslam

#endif  // EVENTOBJECTSLAM_FRAMETRACKER_H