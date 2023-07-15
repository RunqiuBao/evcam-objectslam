#include "camera.h"

#include <limits>
#include "mathutils.h"

#include <logging.h>
TDO_LOGGER("objectslam.camera")


namespace eventobjectslam{

float camera::CameraBase::TriangulateThreeDPointInStereoCamera(const float disparity){
    return _baseline * _kk(0, 0) / disparity;
}


void camera::CameraBase::MatchStereoBBoxes(
    std::vector<TwoDBoundingBox>& leftCamDetections,
    std::vector<TwoDBoundingBox>& rightCamDetections,
    const object::ObjectBase& objectInfo,
    std::vector<std::shared_ptr<TwoDBoundingBox>>& matchedLeftCamDetections,
    std::vector<std::shared_ptr<TwoDBoundingBox>>& matchedRightCamDetections
){
    size_t yMarginForMatch = 20;  //unit is pixel
    float distanceErrorToReject = 2.0;  // unit is m
    std::vector<size_t> indicesMatchedDetectionInRightCam;  // Note: to prevent repeated match to same detection
    for (TwoDBoundingBox& oneDetectionLeftCam : leftCamDetections){
        int indexSmallestDistanceError = -1;
        float smallestDistanceError = std::numeric_limits<float>::max();
        float estimatedDistance = -1.;
        float distanceByScale = objectInfo._templates[objectInfo.indicesInTemplatesArray[oneDetectionLeftCam._templateID]]._simulationCameraInObjectTransform.block(0, 3, 3, 1).norm() / oneDetectionLeftCam._templateScale;
        for (size_t indexDetectionInRightCam=0; indexDetectionInRightCam < rightCamDetections.size(); indexDetectionInRightCam++){
            std::shared_ptr<TwoDBoundingBox> pOneDetectionRightCam = std::make_shared<TwoDBoundingBox>(rightCamDetections[indexDetectionInRightCam]);
            if (std::abs(oneDetectionLeftCam._centerY - (*pOneDetectionRightCam)._centerY) < yMarginForMatch){
                // filter false match by comparing scale distance with triangulation distance
                float distanceByTriangulation = this->TriangulateThreeDPointInStereoCamera(oneDetectionLeftCam._centerX - (*pOneDetectionRightCam)._centerX);
                float distanceError = std::abs(distanceByScale - distanceByTriangulation);
                if (distanceError < distanceErrorToReject && distanceError < smallestDistanceError){
                    indexSmallestDistanceError = indexDetectionInRightCam;
                    smallestDistanceError = distanceError;
                    estimatedDistance = distanceByTriangulation;
                }
            }
        }
        TDO_LOG_DEBUG_FORMAT("indexSmallestDistanceError: %d, smallestDistanceError: %f, estimatedDistance: %f, distanceByScale: %f", indexSmallestDistanceError % smallestDistanceError % estimatedDistance % distanceByScale);
        if (
            indexSmallestDistanceError >= 0 && 
            std::find(indicesMatchedDetectionInRightCam.begin(), indicesMatchedDetectionInRightCam.end(), indexSmallestDistanceError) == indicesMatchedDetectionInRightCam.end()
        ){
            oneDetectionLeftCam.Set3DDistance(estimatedDistance);
            std::shared_ptr<TwoDBoundingBox> matchedLeftCamDetection = std::make_shared<TwoDBoundingBox>(oneDetectionLeftCam);
            std::shared_ptr<TwoDBoundingBox> matchedRightCamDetection = std::make_shared<TwoDBoundingBox>(rightCamDetections[indexSmallestDistanceError]);
            matchedLeftCamDetections.push_back(matchedLeftCamDetection);
            matchedRightCamDetections.push_back(matchedRightCamDetection);
            indicesMatchedDetectionInRightCam.push_back(indexSmallestDistanceError);
        }
    }
    TDO_LOG_DEBUG_FORMAT("Found %d matched detection in this frame.", matchedLeftCamDetections.size());
    return;
}

void camera::CameraBase::CreateThreeDDetections(
    const std::vector<std::shared_ptr<TwoDBoundingBox>>& matchedLeftCamDetections,
    const std::vector<std::shared_ptr<TwoDBoundingBox>>& matchedRightCamDetections,
    const object::ObjectBase& objectInfo,
    std::vector<ThreeDDetection>& threeDDetections
){
    int detectionID = 0;
    float xSize, ySize, zSize;
    xSize = objectInfo._objectExtents[0];
    ySize = objectInfo._objectExtents[1];
    zSize = objectInfo._objectExtents[2];
    Eigen::MatrixXf vertices3D;
    vertices3D.resize(8, 3);
    vertices3D.row(0) << xSize / 2, ySize / 2, zSize / 2;
    vertices3D.row(1) << -xSize / 2, ySize / 2, zSize / 2;
    vertices3D.row(2) << -xSize / 2, -ySize / 2, zSize / 2;
    vertices3D.row(3) << xSize / 2, -ySize / 2, zSize / 2;
    vertices3D.row(4) << xSize / 2, ySize / 2, -zSize / 2;
    vertices3D.row(5) << -xSize / 2, ySize / 2, -zSize / 2;
    vertices3D.row(6) << -xSize / 2, -ySize / 2, -zSize / 2;
    vertices3D.row(7) << xSize / 2, -ySize / 2, -zSize / 2;

    for (const std::shared_ptr<TwoDBoundingBox> matchedLeftCamDetection : matchedLeftCamDetections){
        size_t templateID = (*matchedLeftCamDetection)._templateID;
        Mat44_t objectInCameraTransform = objectInfo._templates[objectInfo.indicesInTemplatesArray[templateID]]._simulationCameraInObjectTransform.inverse();
        objectInCameraTransform.block(0, 3, 3, 1) *= (*matchedLeftCamDetection)._esitmated3DDistance / objectInCameraTransform.block(0, 3, 3, 1).norm(); 
        Eigen::MatrixXf vertices3DInCamera = mathutils::TransformPoints<Eigen::MatrixXf>(objectInCameraTransform, vertices3D);
        ThreeDDetection new3DDetection(objectInCameraTransform, detectionID, vertices3DInCamera, this->_cameraID);
        threeDDetections.push_back(new3DDetection);
        detectionID++;
    }
    TDO_LOG_DEBUG_FORMAT("created %d threeD detections in camera %d", threeDDetections.size() % this->_cameraID);
    return;
}


}  // end of namespace eventobjectslam