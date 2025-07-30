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
    const std::vector<Vec3_t> vertices3DInLandmark,
    const float horizontalSize,
    const std::shared_ptr<object::ObjectBase> pObjectInfo,
    const bool hasFacet
)
:_poseLandmarkInWorld(poseLandmarkInWorld), _vertices3DInLandmark(vertices3DInLandmark), _horizontalSize(horizontalSize), _landmarkID(_nextID++), _pObjectInfo(pObjectInfo), _bIsToDelete(false), _hasFacet(hasFacet)
{
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
        // phase: initialization of this landmark.
        _bestDetectionScore = pRefKeyFrame->_refObjects[idx]->_detection._detectionScore;
    }
    else{
        float incomingDetectionScore = pRefKeyFrame->_refObjects[idx]->_detection._detectionScore;
        if (
            (incomingDetectionScore > _bestDetectionScore)
        ){
            // Note: update landmark pose, if got a better view from the input keyframe.
            Mat44_t poseLandmarkInWorldNew;
            std::vector<Vec3_t> verticesCornersInLandmarkNew;
            ComputeLandmarkPoseInWorldByVertices3D(
                pRefKeyFrame,
                pRefKeyFrame->_refObjects[idx],
                poseLandmarkInWorldNew,
                verticesCornersInLandmarkNew);
            TDO_LOG_INFO_FORMAT("updated landmark (%d), due to better observation (with keyframe %d): detection score change: %f -> %f", _landmarkID % pRefKeyFrame->_keyFrameID % _bestDetectionScore % incomingDetectionScore);
            _bestDetectionScore = incomingDetectionScore;
            SetLandmarkPoseInWorld(poseLandmarkInWorldNew);
            SetVertices3DInLandmark(verticesCornersInLandmarkNew);
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
void LandMark::ComputeLandmarkPoseInWorldByVertices3D(
    const std::shared_ptr<KeyFrame> pRefKeyFrame,
    const std::shared_ptr<RefObject> pRefObjInKeyFrame,
    Mat44_t& poseLandmarkInWorld,
    std::vector<Vec3_t>& vertices3DInLandmark
){
    poseLandmarkInWorld = Eigen::Matrix4f::Identity();
    Vec3_t objCenterInWorld = pRefKeyFrame->GetKeyframePoseInWorld().block<3, 3>(0, 0) * pRefObjInKeyFrame->_detection._objectCenterInRefFrame + pRefKeyFrame->GetKeyframePoseInWorld().col(3).head<3>();
    poseLandmarkInWorld(0, 3) = objCenterInWorld(0);
    poseLandmarkInWorld(1, 3) = objCenterInWorld(1);
    poseLandmarkInWorld(2, 3) = objCenterInWorld(2);
    if (pRefObjInKeyFrame->_detection._hasFacet){
        Vec3_t facetNormalInWorld = pRefKeyFrame->GetKeyframePoseInWorld().block<3, 3>(0, 0) * pRefObjInKeyFrame->_detection._normalOfFacet;
        Mat33_t rotMat;
        GetRotationMatFromVecZ(facetNormalInWorld, rotMat);
        poseLandmarkInWorld.block<3, 3>(0, 0) = rotMat;
    }
    else{
        // if no facet, use keypt1 to align landmark z axis.
        Vec3_t oc2Keypt1 = pRefKeyFrame->GetKeyframePoseInWorld().block<3, 3>(0, 0) * pRefObjInKeyFrame->_detection._vertices3DInRefFrame[0] + pRefKeyFrame->GetKeyframePoseInWorld().col(3).head<3>() - objCenterInWorld;
        Mat33_t rotMat;
        GetRotationMatFromVecZ(oc2Keypt1 / oc2Keypt1.norm(), rotMat);
        poseLandmarkInWorld.block<3, 3>(0, 0) = rotMat;
    }

    int numVertices3DOfRefObj = pRefObjInKeyFrame->_detection._vertices3DInRefFrame.size();
    numVertices3DOfRefObj = (pRefObjInKeyFrame->_detection._hasFacet)?numVertices3DOfRefObj:2;  // Note: 2 is for color cone.
    Mat44_t poseWorldInLandmark = poseLandmarkInWorld.inverse();
    for (int indexCorner=0; indexCorner < numVertices3DOfRefObj; indexCorner++){
        Vec3_t vertexInWorld = pRefKeyFrame->GetKeyframePoseInWorld().block<3, 3>(0, 0) * pRefObjInKeyFrame->_detection._vertices3DInRefFrame[indexCorner] + pRefKeyFrame->GetKeyframePoseInWorld().col(3).head<3>();
        Vec3_t vertexInLandmark = poseWorldInLandmark.block<3, 3>(0, 0) * vertexInWorld + poseWorldInLandmark.col(3).head<3>();
        vertices3DInLandmark.push_back(vertexInLandmark);
    }
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

void LandMark::SetVertices3DInLandmark(const std::vector<Vec3_t>& vertices3DInLandmark) {
    std::lock_guard<std::mutex> lock(_mtxLandmarkSize);
    _vertices3DInLandmark = vertices3DInLandmark;
}

void LandMark::SetLandmarkSize(const float observedHeight, const float horizontalSize) {
    std::lock_guard<std::mutex> lock(_mtxLandmarkSize);
    _horizontalSize = horizontalSize;
    _observedHeight = observedHeight;
}

Vec3_t LandMark::GetOneVertex3DInWorld(size_t indexCorner) const {
    return _poseLandmarkInWorld.block<3, 3>(0, 0) * _vertices3DInLandmark[indexCorner] + _poseLandmarkInWorld.col(3).head<3>();
}

}  // end of namespace eventobjectslam
