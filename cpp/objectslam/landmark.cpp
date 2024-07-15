#include "landmark.h"
#include "object.h"

#include <limits>

#include <logging.h>
TDO_LOGGER("eventobjectslam.landmark")

static float makeCylinderLonger = 1.0;

namespace eventobjectslam {

std::atomic<unsigned int> LandMark::_nextID{0};

LandMark::LandMark(
    const Mat44_t poseLandmarkInWorld,
    const Vec3_t keypt1InLandmark,
    const float horizontalSize,
    const std::shared_ptr<object::ObjectBase> pObjectInfo
)
:_poseLandmarkInWorld(poseLandmarkInWorld), _keypt1InLandmark(keypt1InLandmark * makeCylinderLonger), _horizontalSize(horizontalSize), _landmarkID(_nextID++), _pObjectInfo(pObjectInfo), _bIsToDelete(false)
{
    Vec3_t landmarkCenter = Eigen::Vector3f::Zero(3);
    _vertices3DInLandmark = GetVerticesOf3DBoundingCylinderForObject(4, horizontalSize, landmarkCenter, _keypt1InLandmark);
    _observedHeight = _keypt1InLandmark.norm() * 2;

    _bestDetectionScore = std::numeric_limits<float>::max();
    if (_landmarkID > 200) {
        TDO_LOG_ERROR_FORMAT("too much landmarks initialized (%d)", _landmarkID);
    }
}

bool LandMark::CheckIfObservation(std::shared_ptr<KeyFrame> pRefKeyFrame) {
    std::lock_guard<std::mutex> lock(_mtxObservations);
    return _observations_indices.count(pRefKeyFrame);
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
        float distanceToInputKeyFrame = pRefKeyFrame->_refObjects[idx]->_detection._objectCenterInRefFrame.norm();
        if (
            // CompareDetectionScoreIfBetter("linemod", _bestDetectionScore, pRefKeyFrame->_refObjects[idx]->_detection._detectionScore) &&
            (distanceToInputKeyFrame < _distanceFromBestRefKeyframe)
        ){
            // Note: update landmark pose, if got an closer view from the input keyframe.
            Mat44_t poseLandmarkInWorldNew;
            Vec3_t keypt1InLandmarkNew;
            ComputeLandmarkPoseInWorldAndKeypt1InWolrd(pRefKeyFrame, pRefKeyFrame->_refObjects[idx], poseLandmarkInWorldNew, keypt1InLandmarkNew);
            keypt1InLandmarkNew *= makeCylinderLonger;
            TDO_LOG_INFO_FORMAT("updated landmark (%d), due to betterDistance (with keyframe %d): %f -> %f ; detection score change: %f -> %f", _landmarkID % pRefKeyFrame->_keyFrameID % _distanceFromBestRefKeyframe % distanceToInputKeyFrame % _bestDetectionScore % pRefKeyFrame->_refObjects[idx]->_detection._detectionScore);
            _bestDetectionScore = pRefKeyFrame->_refObjects[idx]->_detection._detectionScore;
            SetLandmarkPoseInWorld(poseLandmarkInWorldNew);
            SetKeypt1InLandmark(keypt1InLandmarkNew);
            SetLandmarkSize(keypt1InLandmarkNew.norm() * 2, pRefKeyFrame->_refObjects[idx]->_detection._horizontalSize);
            _distanceFromBestRefKeyframe = distanceToInputKeyFrame;
        }
    }

    _numObservations++;
}

static void GetRotationMatFromVecZ(
    const Vec3_t vecZ,
    Mat33_t& rotationMat
){
    Vec3_t vecZ_pert = vecZ;
    vecZ_pert(0) += M_PI;
    Vec3_t vecX = vecZ_pert.cross(vecZ);
    vecX = vecX / vecX.norm();
    Vec3_t vecY = vecZ.cross(vecX);
    vecY = vecY / vecY.norm();
    rotationMat.col(0) = vecX;
    rotationMat.col(1) = vecY;
    rotationMat.col(2) = vecZ;
}

// class static method
void LandMark::ComputeLandmarkPoseInWorldAndKeypt1InWolrd(
    const std::shared_ptr<KeyFrame> pRefKeyFrame,
    const std::shared_ptr<RefObject> pRefObjInKeyFrame,
    Mat44_t& poseLandmarkInWorld,
    Vec3_t& keypt1InLandmark
){
    poseLandmarkInWorld = Eigen::Matrix4f::Identity();
    Vec3_t objCenterInWorld = pRefKeyFrame->GetKeyframePoseInWorld().block<3, 3>(0, 0) * pRefObjInKeyFrame->_detection._objectCenterInRefFrame + pRefKeyFrame->GetKeyframePoseInWorld().col(3).head<3>();
    poseLandmarkInWorld(0, 3) = objCenterInWorld(0);
    poseLandmarkInWorld(1, 3) = objCenterInWorld(1);
    poseLandmarkInWorld(2, 3) = objCenterInWorld(2);
    Vec3_t keypt1InWorld = pRefKeyFrame->GetKeyframePoseInWorld().block<3, 3>(0, 0) * pRefObjInKeyFrame->_detection._keypt1InRefFrame + pRefKeyFrame->GetKeyframePoseInWorld().col(3).head<3>();
    Vec3_t oc2Keypt1 = keypt1InWorld - objCenterInWorld;
    Mat33_t rotMat;
    GetRotationMatFromVecZ(oc2Keypt1 / oc2Keypt1.norm(), rotMat);
    poseLandmarkInWorld.block<3, 3>(0, 0) = rotMat;

    keypt1InLandmark = keypt1InLandmark.setZero();
    keypt1InLandmark(2) = oc2Keypt1.norm();  // align landmark z axis with the center2Keypt1 vector.
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

void LandMark::SetKeypt1InLandmark(const Vec3_t& keypt1InLandmark) {
    std::lock_guard<std::mutex> lock(_mtxLandmarkSize);
    _keypt1InLandmark = keypt1InLandmark;
}

void LandMark::SetLandmarkSize(const float observedHeight, const float horizontalSize) {
    std::lock_guard<std::mutex> lock(_mtxLandmarkSize);
    _horizontalSize = horizontalSize;
    _observedHeight = observedHeight;
}

}  // end of namespace eventobjectslam
