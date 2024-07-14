#ifndef MATHUTILS_H
#define MATHUTILS_H

#include <Eigen/Core>
#include <cassert>
#include <opencv2/opencv.hpp>

#include "camera.h"

namespace eventobjectslam {

namespace mathutils {

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

#endif  // MATHUTILS_H