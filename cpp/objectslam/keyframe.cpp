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

void KeyFrame::InitializeObservedLandmarks(std::map<std::shared_ptr<LandMark>, unsigned int> observedLandmarks_indicesRefObj) {
    decltype(_vIdsCorrespLandmarks) vIdsCorrespLandmarks;
    for (auto pObservedLandmark_indexRefObj : observedLandmarks_indicesRefObj) {
        vIdsCorrespLandmarks.push_back(pObservedLandmark_indexRefObj.first->_landmarkID);
    }
    {
        // scope of adding observed landmarks.
        std::lock_guard<std::mutex> lock(_mtxLandmarks);

        _vIdsCorrespLandmarks = vIdsCorrespLandmarks;
        _observedLandmarks_indicesRefObj = observedLandmarks_indicesRefObj;
    }
    // initialize covisibility graph node.
    _graphNode = std::unique_ptr<GraphNode>(new GraphNode(this));

}

} // end of namespace eventobjectslam
