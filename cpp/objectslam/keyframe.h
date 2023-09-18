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

    // ---------- landmark observation management ----------
    void InitializeObservedLandmarks(std::map<std::shared_ptr<LandMark>, unsigned int> observedLandmarks_indicesRefObj);

    void DeleteOneObservedLandmark(std::shared_ptr<LandMark> oneLandmarkToPrune);

    void ReplaceOneObservedLandmark(std::shared_ptr<LandMark> oldLandmark, std::shared_ptr<LandMark> newLandmark);

    // --------- covisibility related methods ---------
    void AddCovisibilityConnection(std::shared_ptr<KeyFrame> pTargetKeyframe, size_t weight);

    std::vector<std::shared_ptr<RefObject>> _refObjects;

    Mat44_t _poseCurrentFrameInWorld;  // Note: pose current to world

    std::shared_ptr<Frame> _pRefFrame;

    unsigned int _keyFrameID;
    static std::atomic<unsigned int> _nextID;

    // observed landmarks
    std::map<std::shared_ptr<LandMark>, unsigned int> _observedLandmarks_indicesRefObj;

    std::map<std::shared_ptr<LandMark>, unsigned int> GetObservedLandmarks() { 
        std::lock_guard<std::mutex> lock(_mtxLandmarks);
        return _observedLandmarks_indicesRefObj;
    }

    // camera instance
    std::shared_ptr<camera::CameraBase> _pCamera;

    bool _bContainNewLandmarks;

private:
    //graph node of the covisiblity graph
    std::unique_ptr<GraphNode> _graphNode = nullptr;

    mutable std::mutex _mtxLandmarks;
    mutable std::mutex _mtxKeyframePose;
    mutable std::mutex _mtxCovisibilityGraph;

};


}  // end of namespace eventobjectslam

#endif  // EVENTOBJECTSLAM_KEYFRAME_H