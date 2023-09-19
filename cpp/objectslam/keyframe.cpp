#include "keyframe.h"

#include <logging.h>
TDO_LOGGER("eventobjectslam.keyframe")

namespace eventobjectslam {

std::atomic<unsigned int> KeyFrame::_nextID{0};

KeyFrame::KeyFrame(const std::shared_ptr<Frame> pRefFrame, const Mat44_t& refKeyFrameInWorldTransform, const std::shared_ptr<camera::CameraBase> pCamera)
:_keyFrameID(_nextID++), _pCamera(pCamera) {
    _poseCurrentFrameInWorld = refKeyFrameInWorldTransform * pRefFrame->GetPose();
    size_t countRefObject = 0;
    for(ThreeDDetection oneDetection : pRefFrame->_threeDDetections){
        std::shared_ptr<RefObject> oneRefObject = std::make_shared<RefObject>(oneDetection, countRefObject);
        _refObjects.push_back(oneRefObject);
        countRefObject++;
    }
    _pRefFrame = pRefFrame;
    _bContainNewLandmarks = false;
}

void KeyFrame::AddCovisibilityConnection(std::shared_ptr<KeyFrame> pTargetKeyframe, size_t weight) {
    std::lock_guard<std::mutex> lock(_mtxCovisibilityGraph);
    _graphNode->AddCovisibilityConnection(pTargetKeyframe, weight);
}

std::vector<std::shared_ptr<KeyFrame>> KeyFrame::GetOrderedCovisibilities() const {
    std::lock_guard<std::mutex> lock(_mtxCovisibilityGraph);
    return _graphNode->GetOrderedCovisibilities();
}

void KeyFrame::InitializeObservedLandmarks(std::map<std::shared_ptr<LandMark>, unsigned int> observedLandmarks_indicesRefObj) {
    {
        // scope of adding observed landmarks.
        std::lock_guard<std::mutex> lock(_mtxLandmarks);
        _observedLandmarks_indicesRefObj = observedLandmarks_indicesRefObj;
    }
    // initialize covisibility graph node.
    _graphNode = std::unique_ptr<GraphNode>(new GraphNode(this));

}

void KeyFrame::DeleteOneObservedLandmark(std::shared_ptr<LandMark> oneLandmarkToPrune) {
    std::lock_guard<std::mutex> lock(_mtxLandmarks);
    _observedLandmarks_indicesRefObj.erase(oneLandmarkToPrune);
    _graphNode->UpdateEraseOneCovisibleLandmark(oneLandmarkToPrune->GetObservations());
}

void KeyFrame::ReplaceOneObservedLandmark(std::shared_ptr<LandMark> oldLandmark, std::shared_ptr<LandMark> newLandmark) {
    std::lock_guard<std::mutex> lock(_mtxLandmarks);
    unsigned int indexRefObj = _observedLandmarks_indicesRefObj[oldLandmark];
    _observedLandmarks_indicesRefObj.erase(oldLandmark);
    _observedLandmarks_indicesRefObj[newLandmark] = indexRefObj;
}

Mat44_t KeyFrame::GetKeyframePoseInWorld() {
    std::lock_guard<std::mutex> lock(_mtxKeyframePose);
    return _poseCurrentFrameInWorld;
}

void KeyFrame::SetKeyframePoseInWorld(const Mat44_t& poseCurrentFrameInWorld) {
    std::lock_guard<std::mutex> lock(_mtxKeyframePose);
    _poseCurrentFrameInWorld = poseCurrentFrameInWorld;
}

} // end of namespace eventobjectslam
