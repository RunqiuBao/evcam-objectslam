#include "camera.h"
#include "mathutils.h"

#include <limits>
#include <opencv2/opencv.hpp>
#include <cassert>

#include <logging.h>
TDO_LOGGER("objectslam.camera")


namespace eventobjectslam{

static const bool CheckTooCloseToImageBorder(
    const float borderDistanceThreshold,
    const float centerX,
    const float centerY,
    const float bWidth,
    const float bHeight,
    const int imgWidth,
    const int imgHeight
){
    TDO_LOG_VERBOSE_FORMAT("borderDistanceThreshold: %f, centerX: %f, centerY: %f, bWidth: %f, bHeight: %f, imgWidth: %f, imgHeight: %f", borderDistanceThreshold % centerX % centerY % bWidth % bHeight % imgWidth % imgHeight);
    if ((centerX - bWidth / 2) < borderDistanceThreshold){
        return true;
    }
    else if ((centerX + bWidth / 2) > (imgWidth - borderDistanceThreshold)){
        return true;
    }
    // Note: do not delete bottom ones
    // else if ((centerY - bHeight / 2) < borderDistanceThreshold) {
    //     return true;
    // }
    else if ((centerY + bHeight / 2) > (imgHeight - borderDistanceThreshold)) {
        return true;
    }
    else {
        return false;
    }
}


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
    const std::vector<TwoDBoundingBox>& leftCamDetections,
    const std::vector<TwoDBoundingBox>& rightCamDetections,
    std::vector<std::shared_ptr<TwoDBoundingBox>>& matchedLeftCamDetections,
    std::vector<std::shared_ptr<TwoDBoundingBox>>& matchedRightCamDetections
){
    float yMarginForMatch  = 20.;  // threshold to reject stereo detection
    float cleanMarginFromImageBorder = 0;
    if (leftCamDetections.size() == 0)
        return;

    const std::shared_ptr<object::ObjectBase> pObjectInfo = leftCamDetections[0]._pObjectInfo;

    for (size_t indexDetection=0; indexDetection < leftCamDetections.size(); indexDetection++){
        std::shared_ptr<TwoDBoundingBox> pOneDetectionLeftCam = std::make_shared<TwoDBoundingBox>(leftCamDetections[indexDetection]);
        std::shared_ptr<TwoDBoundingBox> pOneDetectionRightCam = std::make_shared<TwoDBoundingBox>(rightCamDetections[indexDetection]);
        bool isBadStereo = std::abs(pOneDetectionLeftCam->_centerY - (*pOneDetectionRightCam)._centerY) > yMarginForMatch;
        bool isTooCloseToImageBorder = CheckTooCloseToImageBorder(cleanMarginFromImageBorder, pOneDetectionLeftCam->_centerX, pOneDetectionLeftCam->_centerY, pOneDetectionLeftCam->_bWidth, pOneDetectionLeftCam->_bHeight, _cols, _rows)
                                       || CheckTooCloseToImageBorder(cleanMarginFromImageBorder, pOneDetectionRightCam->_centerX, pOneDetectionRightCam->_centerY, pOneDetectionRightCam->_bWidth, pOneDetectionRightCam->_bHeight, _cols, _rows);
        float depthByTriangulation = this->TriangulateDepthInStereoCamera(pOneDetectionLeftCam->_centerX - pOneDetectionRightCam->_centerX);
        TDO_LOG_DEBUG_FORMAT("detection (%f), isBadStereo %s, isTooCloseToImageBorder %s", indexDetection % (isBadStereo?"True":"False") % (isTooCloseToImageBorder?"True":"False"));
        if (!isBadStereo && !isTooCloseToImageBorder){            
            pOneDetectionLeftCam->Set3DDepth(depthByTriangulation);
            pOneDetectionRightCam->Set3DDepth(depthByTriangulation);
            matchedLeftCamDetections.push_back(pOneDetectionLeftCam);
            matchedRightCamDetections.push_back(pOneDetectionRightCam);
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
    std::vector<Vec3_t> threeDPoints;
    std::vector<ThreeDDetection> threeDDetectionCandidates;
    for (size_t ii = 0; ii < matchedLeftCamDetections.size(); ii++){
        const std::shared_ptr<TwoDBoundingBox> pMatchedLeftCamDetection = matchedLeftCamDetections[ii];
        const std::shared_ptr<TwoDBoundingBox> pMatchedRightCamDetection = matchedRightCamDetections[ii];
        float X, Y;
        this->ProjectPointTo3DByDepth((*pMatchedLeftCamDetection)._esitmated3DDepth, (*pMatchedLeftCamDetection)._centerX, (*pMatchedLeftCamDetection)._centerY, X, Y);
        Vec3_t objectCenterInRefFrame(X, Y, (*pMatchedLeftCamDetection)._esitmated3DDepth);
        // get keypt1 in 3D
        float keypt1_x = (*pMatchedLeftCamDetection)._keypts[0][0];
        float keypt1_y = (*pMatchedLeftCamDetection)._keypts[0][1];
        this->ProjectPointTo3DByDepth((*pMatchedLeftCamDetection)._esitmated3DDepth, keypt1_x, keypt1_y, X, Y);
        Vec3_t keypt1InCam(X, Y, pMatchedLeftCamDetection->_esitmated3DDepth);
        float horizontalSize = pMatchedLeftCamDetection->_bWidth * pMatchedLeftCamDetection->_esitmated3DDepth / _kk(0, 0);

        ThreeDDetection new3DDetection(
            objectCenterInRefFrame,
            detectionID,
            this->_cameraID,
            pMatchedLeftCamDetection->_detectionScore,
            pMatchedLeftCamDetection,
            pMatchedRightCamDetection,
            keypt1InCam,
            horizontalSize,
            pMatchedLeftCamDetection->_pObjectInfo
        );
        threeDDetectionCandidates.push_back(new3DDetection);
        threeDPoints.push_back(objectCenterInRefFrame);
        detectionID++;
    }
    // filter non-plane 3d detections
    std::vector<int> indicesThreeDPoints;
    mathutils::FilterNonPlanePoints(threeDPoints, 0.2, indicesThreeDPoints);
    for (const int indexPoint : indicesThreeDPoints){
        threeDDetections.push_back(threeDDetectionCandidates[indexPoint]);
    }
    TDO_LOG_DEBUG_FORMAT("created %d threeD detections in camera %d", threeDDetections.size() % this->_cameraID);
    return;
}

void camera::CameraBase::ProjectPoints(
    const Eigen::Ref<const Eigen::Matrix<float, 3, Eigen::Dynamic>> points,  // Note: stride<3, 1> is verified correct by getting toprows(3) from a 4*5 matrix.
    Eigen::Ref<Eigen::Matrix<float, 2, Eigen::Dynamic>> dstPoints
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
        dstPoints = Eigen::Map<Eigen::Matrix<float, 2, Eigen::Dynamic>>(reinterpret_cast<float*>(cvDst.data), 2, N);
    }
}


}  // end of namespace eventobjectslam
