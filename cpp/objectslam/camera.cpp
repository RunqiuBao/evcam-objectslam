#include "camera.h"
#include "mathutils.h"

#include <limits>
#include <opencv2/opencv.hpp>
#include <cassert>

#include <logging.h>
TDO_LOGGER("objectslam.camera")


namespace eventobjectslam{

void camera::CameraBase::ProjectPointTo3DByDepth(const float Z, const float u, const float v, float& X, float& Y){
    X = (u - _kk(0, 2)) /_kk(0, 0) * Z;
    Y = (v - _kk(1, 2)) / _kk(1, 1) * Z;
    return;
}

float camera::CameraBase::TriangulateDepthInStereoCamera(const float disparity){
    // ref: https://docs.opencv.org/4.6.0/dd/d53/tutorial_py_depthmap.html
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
        int indexsmallestDepthError = -1;
        float smallestDepthError = std::numeric_limits<float>::max();
        float estimatedDepth = -1.;
        float depthByScale = objectInfo._templates[objectInfo.indicesInTemplatesArray[oneDetectionLeftCam._templateID]]._simulationCameraInObjectTransform.block(0, 3, 3, 1).norm() / oneDetectionLeftCam._templateScale;
        for (size_t indexDetectionInRightCam=0; indexDetectionInRightCam < rightCamDetections.size(); indexDetectionInRightCam++){
            std::shared_ptr<TwoDBoundingBox> pOneDetectionRightCam = std::make_shared<TwoDBoundingBox>(rightCamDetections[indexDetectionInRightCam]);
            if (std::abs(oneDetectionLeftCam._centerY - (*pOneDetectionRightCam)._centerY) < yMarginForMatch){
                // filter false match by comparing scale distance with triangulation distance
                float depthByTriangulation = this->TriangulateDepthInStereoCamera(oneDetectionLeftCam._centerX - (*pOneDetectionRightCam)._centerX);
                float distanceError = std::abs(depthByScale - depthByTriangulation);
                if (distanceError < distanceErrorToReject && distanceError < smallestDepthError){
                    indexsmallestDepthError = indexDetectionInRightCam;
                    smallestDepthError = distanceError;
                    estimatedDepth = depthByTriangulation;
                }
            }
        }
        TDO_LOG_DEBUG_FORMAT("indexsmallestDepthError: %d, smallestDepthError: %f, estimatedDepth: %f, depthByScale: %f", indexsmallestDepthError % smallestDepthError % estimatedDepth % depthByScale);
        if (
            indexsmallestDepthError >= 0 && 
            std::find(indicesMatchedDetectionInRightCam.begin(), indicesMatchedDetectionInRightCam.end(), indexsmallestDepthError) == indicesMatchedDetectionInRightCam.end()
        ){
            oneDetectionLeftCam.Set3DDepth(estimatedDepth);
            std::shared_ptr<TwoDBoundingBox> matchedLeftCamDetection = std::make_shared<TwoDBoundingBox>(oneDetectionLeftCam);
            std::shared_ptr<TwoDBoundingBox> matchedRightCamDetection = std::make_shared<TwoDBoundingBox>(rightCamDetections[indexsmallestDepthError]);
            matchedLeftCamDetections.push_back(matchedLeftCamDetection);
            matchedRightCamDetections.push_back(matchedRightCamDetection);
            indicesMatchedDetectionInRightCam.push_back(indexsmallestDepthError);
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
    vertices3D.resize(3, 8);
    vertices3D.col(0) << xSize / 2, ySize / 2, zSize / 2;
    vertices3D.col(1) << -xSize / 2, ySize / 2, zSize / 2;
    vertices3D.col(2) << -xSize / 2, -ySize / 2, zSize / 2;
    vertices3D.col(3) << xSize / 2, -ySize / 2, zSize / 2;
    vertices3D.col(4) << xSize / 2, ySize / 2, -zSize / 2;
    vertices3D.col(5) << -xSize / 2, ySize / 2, -zSize / 2;
    vertices3D.col(6) << -xSize / 2, -ySize / 2, -zSize / 2;
    vertices3D.col(7) << xSize / 2, -ySize / 2, -zSize / 2;

    for (const std::shared_ptr<TwoDBoundingBox> matchedLeftCamDetection : matchedLeftCamDetections){
        size_t templateID = (*matchedLeftCamDetection)._templateID;
        Mat44_t objectInCameraTransform = objectInfo._templates[objectInfo.indicesInTemplatesArray[templateID]]._simulationCameraInObjectTransform.inverse();
        float X, Y;
        this->ProjectPointTo3DByDepth((*matchedLeftCamDetection)._esitmated3DDepth, (*matchedLeftCamDetection)._centerX, (*matchedLeftCamDetection)._centerY, X, Y);
        objectInCameraTransform(0, 3) = X;
        objectInCameraTransform(1, 3) = Y;
        objectInCameraTransform(2, 3) = (*matchedLeftCamDetection)._esitmated3DDepth;
        Eigen::MatrixXf vertices3DInCamera = mathutils::TransformPoints<Eigen::MatrixXf>(objectInCameraTransform, vertices3D);
        ThreeDDetection new3DDetection(objectInCameraTransform, detectionID, vertices3DInCamera, this->_cameraID);
        threeDDetections.push_back(new3DDetection);
        detectionID++;
    }
    TDO_LOG_DEBUG_FORMAT("created %d threeD detections in camera %d", threeDDetections.size() % this->_cameraID);
    return;
}

void camera::CameraBase::ProjectPoints(
    const Eigen::Ref<const Eigen::Matrix<float, 3, Eigen::Dynamic>, 0, Eigen::Stride<3, 1>> points,  // Note: stride<3, 1> is verified correct by getting toprows(3) from a 4*5 matrix.
    Eigen::Ref<Eigen::Matrix<float, 2, Eigen::Dynamic>, 0, Eigen::Stride<2, 1> > dstPoints
){
    const size_t N = points.cols();
    // Allocate dstPoints if it is not preallocated
    if (dstPoints.cols() != N) {
        dstPoints = Eigen::Matrix<float, 2, Eigen::Dynamic>(2, N);
    }
    const cv::Mat cvSrc(N, 1, CV_32FC3, const_cast<float*>(points.data()));
    const cv::Mat& cvKK = cv::Mat(3, 3, CV_32FC1, const_cast<float*>(_kk.data())).t();  // Eigen column major gets transposed in OpenCV (row major)
    const cv::Mat& rvec = cv::Mat::zeros(1, 3, CV_32FC1);
    const cv::Mat& tvec = cv::Mat::zeros(1, 3, CV_32FC1);
    cv::Mat cvDst(N, 1, CV_32FC2, dstPoints.data());
    cv::projectPoints(cvSrc, rvec, tvec, cvKK, _distortCoef, cvDst);
    if ( dstPoints.data() != reinterpret_cast<float*>(cvDst.data) ) {
        dstPoints = Eigen::Map<Eigen::Matrix<float, 2, Eigen::Dynamic>, 0, Eigen::Stride<2, 1> >(reinterpret_cast<float*>(cvDst.data), 2, N);
    }
}


}  // end of namespace eventobjectslam