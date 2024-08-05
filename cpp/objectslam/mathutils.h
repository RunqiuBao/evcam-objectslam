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
std::vector<cv::Point> ProjectPoints3DToPoints2D(Eigen::MatrixXf& mPoints3D, camera::CameraBase& camera);

cv::Mat Draw2DHullMaskFrom2DPointsSet(const std::vector<cv::Point>& points2DCV, const size_t imageH, const size_t imageW);

Eigen::Vector4f CreateQuatRotateDirection(const Eigen::Vector3f sourceDir, const Eigen::Vector3f targetDir);

Eigen::Vector4f ConvertQuatFromAxisAngle(const Eigen::Vector3f axis, const float angle);

Eigen::Matrix4f ConvertMatrixFromQuat(const Eigen::Vector4f quat);

std::string FillZeros(const std::string& str, const int width);

std::vector<size_t> GetListOfRandomIndex(const size_t iStart, const size_t iEnd, const size_t numElements);

} // end of mathutils

} // end of eventobjectslam
