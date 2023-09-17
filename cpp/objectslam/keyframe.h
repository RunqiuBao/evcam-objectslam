#ifndef EVENTOBJECTSLAM_KEYFRAME_H
#define EVENTOBJECTSLAM_KEYFRAME_H

#include "frame.h"
#include "object.h"
#include "landmark.h"
#include "camera.h"
#include "graphnode.h"

#include <opencv2/opencv.hpp>

namespace eventobjectslam{

class Frame;  // Note: due to mutual reference.
class LandMark;  // Note: due to mutual reference.
class GraphNode;  // Note: due to mutual reference.

class KeyFrame {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // Note: use aligned eigen matrix in following member functions

    // operator overrides
    bool operator==(const KeyFrame& oneKeyframe) const { return _keyFrameID == oneKeyframe._keyFrameID; }
    bool operator!=(const KeyFrame& oneKeyframe) const { return !(*this == oneKeyframe); }

    KeyFrame(const std::shared_ptr<Frame> pRefFrame, const Mat44_t& refKeyFrameInWorldTransform, const std::shared_ptr<camera::CameraBase> pCamera);

    std::vector<std::shared_ptr<RefObject>> _refObjects;
    std::vector<int> _vIdsCorrespLandmarks;  // Note: same size as _refObjects

    Mat44_t _poseCurrentFrameInWorld;  // Note: pose current to world

    std::shared_ptr<Frame> _pRefFrame;

    unsigned int _keyFrameID;
    static std::atomic<unsigned int> _nextID;

    // observed landmarks
    std::vector<std::shared_ptr<LandMark>> _observedLandmarks;

    std::vector<std::shared_ptr<LandMark>> GetLandmarks() { 
        std::lock_guard<std::mutex> lock(_mtxLandmarks);
        return _observedLandmarks; 
    }

    // camera instance
    std::shared_ptr<camera::CameraBase> _pCamera;

    //graph node of the covisiblity graph
    const std::unique_ptr<GraphNode> _graphNode = nullptr;

private:
    mutable std::mutex _mtxLandmarks;
    mutable std::mutex _mtxKeyframePose;

};


}  // end of namespace eventobjectslam

#endif  // EVENTOBJECTSLAM_KEYFRAME_H