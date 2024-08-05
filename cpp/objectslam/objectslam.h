#ifndef EVENTOBJECTSLAM_H
#define EVENTOBJECTSLAM_H

#include <iostream>
#include <stdexcept>
#undef eigen_assert
#define eigen_assert(x) do { \
    if (!(x)) { \
        std::cerr << "Eigen assertion failed: " #x << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        /* Take other actions if needed, e.g., set an error flag, etc. */ \
    } \
} while (0)

#include <Eigen/Core>
#include <Eigen/Dense>

#include <memory>

namespace eventobjectslam {

template<size_t R>
using VecR_d = Eigen::Matrix<double, R, 1>;

template<size_t R, size_t C>
using MatRC_d = Eigen::Matrix<double, R, C>;

typedef Eigen::Matrix4f Mat44_t;
typedef Eigen::Matrix3f Mat33_t;
using Mat33_d = Eigen::Matrix3d;
using Mat44_d = Eigen::Matrix4d;

typedef Eigen::Vector3f Vec3_t;
typedef Eigen::Vector2f Vec2_t;
typedef Eigen::Vector3d Vec3_d;
typedef VecR_d<5> Vec5_d;
typedef VecR_d<4> Vec4_d;
typedef VecR_d<6> Vec6_d;
typedef VecR_d<7> Vec7_d;

typedef std::array<float, 3> ObjectExtents;

}

#endif  // EVENTOBJECTSLAM_H
