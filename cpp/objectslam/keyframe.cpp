#include "keyframe.h"

#include <set>
#include <algorithm>
#include <cassert>
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
    if (!_graphNode) {
        return;  // Note: covisibility graph not initialized yet (keyframe still being created).
    }
    _graphNode->AddCovisibilityConnection(pTargetKeyframe, weight);
}

void KeyFrame::DeleteCovisibilityConnection(std::shared_ptr<KeyFrame> pTargetKeyframe) {
    std::lock_guard<std::mutex> lock(_mtxCovisibilityGraph);
    if (!_graphNode) {
        return;  // Note: covisibility graph not initialized yet (keyframe still being created).
    }
    _graphNode->DeleteCovisibilityConnection(pTargetKeyframe);
}

std::vector<std::shared_ptr<KeyFrame>> KeyFrame::GetOrderedCovisibilities() const {
    std::lock_guard<std::mutex> lock(_mtxCovisibilityGraph);
    if (!_graphNode) {
        return {};  // Note: covisibility graph not initialized yet (keyframe still being created).
    }
    return _graphNode->GetOrderedCovisibilities();
}

std::vector<std::shared_ptr<KeyFrame>> KeyFrame::GetOrderedFullCovisibilities() const {
    std::lock_guard<std::mutex> lock(_mtxCovisibilityGraph);
    if (!_graphNode) {
        return {};  // Note: covisibility graph not initialized yet (keyframe still being created).
    }
    std::vector<std::shared_ptr<KeyFrame>> vFullCovisibilities = _graphNode->GetOrderedCovisibilities();
    std::set<unsigned int> profileOfObservedLandmarks = GetProfileOfObservedLandmarks();
    for (auto pFullCovisible : vFullCovisibilities) {
        if (!pFullCovisible) {
            continue; // empty pointer
        }
        auto theOtherProfile = pFullCovisible->GetProfileOfObservedLandmarks();
        if (
            profileOfObservedLandmarks == theOtherProfile
            || std::includes(profileOfObservedLandmarks.begin(), profileOfObservedLandmarks.end(), theOtherProfile.begin(), theOtherProfile.end())     
        ) {
            vFullCovisibilities.push_back(pFullCovisible);
        }
    }
    return vFullCovisibilities;
}

std::set<unsigned int> KeyFrame::GetProfileOfObservedLandmarks() const {
    std::lock_guard<std::mutex> lock(_mtxLandmarks);
    std::set<unsigned int> profileObservedLandmarks;
    for (auto pObservedLandmark_indexRefObj: _observedLandmarks_indicesRefObj) {
        profileObservedLandmarks.insert(pObservedLandmark_indexRefObj.first->_landmarkID);
    }
    return profileObservedLandmarks;
}

void KeyFrame::InitializeObservedLandmarks(std::map<std::shared_ptr<LandMark>, unsigned int> observedLandmarks_indicesRefObj) {
    {
        // scope of adding observed landmarks.
        std::lock_guard<std::mutex> lock(_mtxLandmarks);
        _observedLandmarks_indicesRefObj = observedLandmarks_indicesRefObj;
    }
    // initialize covisibility graph node.
    {
        std::lock_guard<std::mutex> lock(_mtxCovisibilityGraph);
        _graphNode = std::unique_ptr<GraphNode>(new GraphNode(this));
    }
    _graphNode->ComputeCovisibility();  // Note: initialize the node in covisibility graph.

}

void KeyFrame::DeleteOneObservedLandmark(std::shared_ptr<LandMark> oneLandmarkToPrune) {
    std::lock_guard<std::mutex> lock(_mtxLandmarks);
    _observedLandmarks_indicesRefObj.erase(oneLandmarkToPrune);
    // Note: the covisibility graph node is only created at the end of CreateNewLandmarks; a prune/merge from
    // the mapper thread can reach a keyframe that is still being created — guard against the null graph node.
    // (lock order: _mtxLandmarks -> _mtxCovisibilityGraph; no path takes them in the reverse order.)
    std::lock_guard<std::mutex> lockGraph(_mtxCovisibilityGraph);
    if (_graphNode) {
        _graphNode->UpdateEraseOneCovisibleLandmark(oneLandmarkToPrune->GetObservations());
    }
}

void KeyFrame::ReplaceOneObservedLandmark(std::shared_ptr<LandMark> oldLandmark, std::shared_ptr<LandMark> newLandmark) {
    std::lock_guard<std::mutex> lock(_mtxLandmarks);
    // A keyframe observes each object instance at most once (exclusive association). If this keyframe already
    // observes the merge target (two co-observed instances got merged), replacing would overwrite the target's
    // entry and corrupt the map — instead just drop the old entry and keep the existing target association.
    if (_observedLandmarks_indicesRefObj.count(newLandmark) > 0) {
        TDO_LOG_WARN_FORMAT("keyframe(%d) already observes merge-target landmark(%d); dropping its entry for merged landmark(%d).",
                                _keyFrameID % newLandmark->_landmarkID % oldLandmark->_landmarkID);
        _observedLandmarks_indicesRefObj.erase(oldLandmark);
        return;
    }
    if (_observedLandmarks_indicesRefObj.count(oldLandmark) == 0) {
        // Note: the observation may have been removed concurrently (e.g. pruning); nothing to replace.
        return;
    }
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
