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
:_poseLandmarkInWorld(poseLandmarkInWorld), _landmarkID(_nextID++), _pObjectInfo(pObjectInfo){
    _vertices3DInLandmark = GetVerticesOf3DBoundingBoxFromObject(pObjectInfo);
    _bestDetectionScore = std::numeric_limits<float>::max();
}

void LandMark::AddObservation(std::shared_ptr<KeyFrame> pRefKeyFrame, unsigned int idx){
    std::lock_guard<std::mutex> lock(_mtxObservations);
    if (_observations_indices.count(pRefKeyFrame)) {
        return;
    }
    _observations_indices[pRefKeyFrame] = idx;
    if (CompareDetectionScoreIfBetter("linemod", _bestDetectionScore, pRefKeyFrame->_refObjects[idx]->_detection._detectionScore)){
        // Note: update landmark orientation.
        Mat44_t poseLandmarkInWorldNew = pRefKeyFrame->_poseCurrentFrameInWorld * pRefKeyFrame->_refObjects[idx]->_detection._objectInCameraTransform;
        _poseLandmarkInWorld.block(0, 0, 3, 3) = poseLandmarkInWorldNew.block(0, 0, 3, 3);
    }
    _numObservations++;
}

}  // end of namespace eventobjectslam
