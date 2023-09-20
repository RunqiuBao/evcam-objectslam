#include "landmark.h"
#include "object.h"

#include <limits>

#include <logging.h>
TDO_LOGGER("eventobjectslam.landmark")

namespace eventobjectslam {

std::atomic<unsigned int> LandMark::_nextID{0};

LandMark::LandMark(
    const Mat44_t poseLandmarkInWorld,
    const std::shared_ptr<object::ObjectBase> pObjectInfo
)
:_poseLandmarkInWorld(poseLandmarkInWorld), _landmarkID(_nextID++), _pObjectInfo(pObjectInfo), _bIsToDelete(false) {
    _vertices3DInLandmark = GetVerticesOf3DBoundingBoxFromObject(pObjectInfo);
    _bestDetectionScore = std::numeric_limits<float>::max();
    if (_landmarkID > 100) {
        TDO_LOG_CRITICAL_FORMAT("iiregular landmarkID(%d)", _landmarkID);
    }
}

void LandMark::AddObservation(std::shared_ptr<KeyFrame> pRefKeyFrame, unsigned int idx){
    std::lock_guard<std::mutex> lock(_mtxObservations);
    if (_observations_indices.count(pRefKeyFrame)) {
        return;
    }
    _observations_indices[pRefKeyFrame] = idx;
    if (_observations_indices.size() == 1){
        _bestDetectionScore = pRefKeyFrame->_refObjects[idx]->_detection._detectionScore;
        _distanceFromBestRefKeyframe = (_poseLandmarkInWorld.block(0, 3, 3, 1) - pRefKeyFrame->GetKeyframePoseInWorld().block(0, 3, 3, 1)).norm();
    }
    else{
        float distanceToInputKeyFrame = pRefKeyFrame->_refObjects[idx]->_detection._objectInCameraTransform.block(0, 3, 3, 1).norm();
        if (
            // CompareDetectionScoreIfBetter("linemod", _bestDetectionScore, pRefKeyFrame->_refObjects[idx]->_detection._detectionScore) &&
            (distanceToInputKeyFrame < _distanceFromBestRefKeyframe)
        ){
            // Note: update landmark pose, if got an closer view from the input keyframe.
            Mat44_t poseLandmarkInWorldNew = pRefKeyFrame->GetKeyframePoseInWorld() * pRefKeyFrame->_refObjects[idx]->_detection._objectInCameraTransform;
            TDO_LOG_INFO_FORMAT("updated landmark (%d), due to betterDistance (with keyframe %d): %f -> %f ; linemod score change: %f -> %f", _landmarkID % pRefKeyFrame->_keyFrameID % _distanceFromBestRefKeyframe % distanceToInputKeyFrame % _bestDetectionScore % pRefKeyFrame->_refObjects[idx]->_detection._detectionScore);
            _bestDetectionScore = pRefKeyFrame->_refObjects[idx]->_detection._detectionScore;
            SetLandmarkPoseInWorld(poseLandmarkInWorldNew);
            _distanceFromBestRefKeyframe = distanceToInputKeyFrame;
        }
    }

    _numObservations++;
}

void LandMark::DeleteObservation(std::shared_ptr<KeyFrame> pRefKeyFrame) {
    if (!_observations_indices.count(pRefKeyFrame)) {
        return;
    }
    std::lock_guard<std::mutex> lock(_mtxObservations);
    _observations_indices.erase(pRefKeyFrame);
}

Mat44_t LandMark::GetLandmarkPoseInWorld() {
    std::lock_guard<std::mutex> lock(_mtxLandmarkPose);
    return _poseLandmarkInWorld;
}

void LandMark::SetLandmarkPoseInWorld(const Mat44_t& poseLandmarkInWorld) {
    std::lock_guard<std::mutex> lock(_mtxLandmarkPose);
    _poseLandmarkInWorld = poseLandmarkInWorld;  // Note: for Eigen matrix, `=` is deep copy.
}

}  // end of namespace eventobjectslam
