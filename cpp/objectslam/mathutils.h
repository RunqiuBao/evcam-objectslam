#pragma once

#include <stdexcept>
#include <cassert>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include <pcl/point_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#include "camera.h"

namespace eventobjectslam {

namespace mathutils {

template<typename ValueType>
Eigen::Matrix<ValueType, 4, 4> FixRxRzAndY(const Eigen::Matrix<ValueType, 4, 4>& pose) {
    std::cout << "Input pose: \n" << pose << std::endl;
    Eigen::Matrix<ValueType, 4, 4> fixedPose = pose;
    Eigen::Matrix<ValueType, 3, 3> rotMat = fixedPose.template block<3, 3>(0, 0);
    Eigen::Matrix<ValueType, 3, 1> euler = rotMat.eulerAngles(1, 2, 0);
    ValueType Ry = euler[0];
    if (Ry > M_PI / 2) {
        Ry -= M_PI;
    }
    Eigen::AngleAxis<ValueType> aaRy(Ry, Eigen::Matrix<ValueType, 3, 1>::UnitY());
    fixedPose.template block<3, 3>(0, 0) = aaRy.toRotationMatrix();
    fixedPose(1, 3) = 0;
    std::cout << "eulerY: " << Ry << ", output pose: \n" << fixedPose << std::endl;
    return fixedPose;
}

void EstimatePlaneFromPoints(
    const std::vector<Vec3_t> points,
    const float planeDistanceThreshold,
    pcl::ModelCoefficients::Ptr& planeCoeff,
    pcl::PointIndices::Ptr& pIndicesInliers
);

float ComputeDistanceFromPlane(const pcl::ModelCoefficients::Ptr plane_coefficients, const pcl::PointXYZ& query_point);

Mat33_t GetRotationMatrixFromVectors(const Vec3_t& vectorA, const Vec3_t& vectorB);

template<typename MatrixType>
MatrixType TransformPoints(const MatrixType& transform, const MatrixType& points){
    // Note: points should be n*3 or 3*n shape, transform is 4*4 Eigen matrix
    assert(transform.rows() == 4 && transform.cols() == 4);
    assert(points.rows() == 3 || points.cols() == 3);
    if (points.rows() == 3){
        return (transform.block(0, 0, 3, 3) * points).colwise() + transform.block(0, 3, 3, 1).col(0);
    }
    else{
        return (transform.block(0, 0, 3, 3) * points.transpose()).colwise() + transform.block(0, 3, 3, 1).col(0);
    }
}

template<typename T>
T RotateAroundPrimeAxis(const double theta, const std::string axisName){
    T rotMat = T::Identity();
    if (axisName == "x"){
        rotMat << 1, 0, 0,
                  0, std::cos(theta), -std::sin(theta),
                  0, std::sin(theta), std::cos(theta);
    }
    else if (axisName == "y"){
        rotMat << std::cos(theta), 0, std::sin(theta),
                  0, 1, 0,
                  -std::sin(theta), 0, std::cos(theta);
    }
    else if (axisName == "z"){
        rotMat << std::cos(theta), -std::sin(theta), 0,
                  std::sin(theta), std::cos(theta), 0,
                  0, 0, 1; 
    }
    else{
        throw std::invalid_argument("Invalid axisName: " + axisName);
    }
    return rotMat;
}


void FilterNonPlanePoints(
    const std::vector<Vec3_t> points,
    const float planeDistanceThreshold,
    std::vector<int>& indicesPoints
);

/**
 *   mPoints3D: 3xn Eigen matrix
 * 
**/
std::vector<cv::Point> ProjectPoints3DToPoints2D(const Eigen::MatrixXf& mPoints3D, camera::CameraBase& camera);

cv::Mat Draw2DHullMaskFrom2DPointsSet(const std::vector<cv::Point>& points2DCV, const size_t imageH, const size_t imageW);

Eigen::Vector4f CreateQuatRotateDirection(const Eigen::Vector3f sourceDir, const Eigen::Vector3f targetDir);

Eigen::Vector4f ConvertQuatFromAxisAngle(const Eigen::Vector3f axis, const float angle);

Eigen::Matrix4f ConvertMatrixFromQuat(const Eigen::Vector4f quat);

std::string FillZeros(const std::string& str, const int width);

std::vector<size_t> GetListOfRandomIndex(const size_t iStart, const size_t iEnd, const size_t numElements);


// ==================== Two view recontruction methods ====================
int CheckRT(const Eigen::Matrix3f& R, const Eigen::Vector3f& t, const std::vector<cv::Point2f> &vKeys1, const std::vector<cv::Point2f> &vKeys2,
            const Eigen::Matrix3f& K, std::vector<cv::Point3f>& vP3D, float th2, std::vector<bool>& vbGood, float& parallax);

bool ReconstructH(
    int numMatchedPoints,
    Eigen::Matrix3f& H21,
    const Eigen::Matrix3f& K,
    Eigen::Matrix4f& T21,
    std::vector<cv::Point2f>& vKeyPts1,
    std::vector<cv::Point2f>& vKeyPts2,
    std::vector<cv::Point3f>& vP3D,
    std::vector<bool> &vbTriangulated,
    float minParallax,
    int minTriangulated
);

/**
 * args: 
 *      
 */
bool TrackWithHomography(
    std::vector<cv::Point2f>& srcPoints2D,
    std::vector<cv::Point2f>& dstPoints2D,
    std::vector<cv::Point3f>& currPoints3D,
    const Eigen::Matrix3f& cameraMatrix,
    Mat44_t& currFrameInRefKeyFrame
);

void RestoreTranslationScale(
    Eigen::Matrix4f& T21,
    std::vector<cv::Point2f>& c1Points2D,
    std::vector<cv::Point2f>& c2Points2D,
    std::vector<cv::Point3f>& c1Points3D,
    const Eigen::Matrix3f& K
);

} // end of mathutils

} // end of eventobjectslam
