#ifndef EVENTOBJECTSLAM_H
#define EVENTOBJECTSLAM_H

#include <iostream>
#include <stdexcept>
// Note: overwrite eigen_assert to prevent it aborting running program
#undef eigen_assert
#define eigen_assert(x) do { \
    if (!(x)) { \
        std::cerr << "Eigen assertion failed: " #x << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    } \
} while (0)

#include <Eigen/Core>
#include <Eigen/Dense>

#include <memory>

namespace eventobjectslam {

typedef Eigen::Matrix4f Mat44_t;
typedef Eigen::Matrix3f Mat33_t;
typedef Eigen::Vector3f Vec3_t;
typedef Eigen::Vector2f Vec2_t;
typedef Eigen::Vector3d Vec3_d;
typedef Eigen::Vector2d Vec2_d;
typedef Eigen::Matrix<double, 5, 1> Vec5_d;
typedef std::array<float, 3> ObjectExtents;
typedef Eigen::Matrix3Xd Mat3Xd; // 3 rows, N cols matrix.
typedef Eigen::Matrix2Xd Mat2Xd;

// -------- double precision, mainly used in g2outils -----------
template<size_t R, size_t C>
using MatRC_d = Eigen::Matrix<double, R, C>;

using Mat33_d = Eigen::Matrix3d;
using Mat44_d = Eigen::Matrix4d;
using Mat55_d = Eigen::Matrix<double, 5, 5>;

template<size_t R>
using VecR_d = Eigen::Matrix<double, R, 1>;

using Vec3_d = Eigen::Vector3d;
using Vec4_d = VecR_d<4>;
using Vec6_d = VecR_d<6>;
using Vec7_d = VecR_d<7>;
using Vec8_d = VecR_d<8>;
using Vec9_d = VecR_d<9>;

}

#endif  // EVENTOBJECTSLAM_H
