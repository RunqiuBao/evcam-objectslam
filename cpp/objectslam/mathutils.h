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

/*--------------------------------- templated utils ---------------------------------------*/
template <class T>
Eigen::Quaternion<T> zyx_euler_to_quat(const T& roll, const T& pitch, const T& yaw){  // Note: all angle in radian.
    T sy = std::sin(yaw * 0.5);
    T cy = std::cos(yaw * 0.5);
    T sp = std::sin(pitch * 0.5);
    T cp = std::cos(pitch * 0.5);
    T sr = std::sin(roll * 0.5);
    T cr = std::cos(roll * 0.5);
    T w = cr * cp * cy + sr * sp * sy;
    T x = sr * cp * cy - cr * sp * sy;
    T y = cr * sp * cy + sr * cp * sy;
    T z = cr * cp * sy - sr * sp * cy;
    return Eigen::Quaternion<T>(w, x, y, z);
}

template Eigen::Quaterniond zyx_euler_to_quat<double>(const double&, const double&, const double&);
template Eigen::Quaternionf zyx_euler_to_quat<float>(const float&, const float&, const float&);

// input is (3, N) shape points (or (2, N) for 2D), output (4, n) shape (or (3, N) for 2D).
template <class T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> real_to_homo_coord(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& input_pts)
{
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> output_pts_homo;
    int dim_pts = input_pts.rows();
    int num_pts = input_pts.cols();

    output_pts_homo.resize(dim_pts + 1, num_pts);
    output_pts_homo << input_pts,
                       Eigen::Matrix<T, 1, Eigen::Dynamic>::Ones(num_pts);
    return output_pts_homo;
}
template Eigen::MatrixXd real_to_homo_coord<double>(const Eigen::MatrixXd&);
template Eigen::MatrixXf real_to_homo_coord<float>(const Eigen::MatrixXf&);

template <class T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> homo_to_real_coord(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& input_pts_homo)
{
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> output_pts(input_pts_homo.rows() - 1, input_pts_homo.cols());
    for (int ii = 0; ii < input_pts_homo.rows() - 1; ii++){
        output_pts.row(ii) = input_pts_homo.row(ii).array() / input_pts_homo.bottomRows(1).array();
    }
    return output_pts;
}
template Eigen::MatrixXd homo_to_real_coord<double>(const Eigen::MatrixXd&);
template Eigen::MatrixXf homo_to_real_coord<float>(const Eigen::MatrixXf&);



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