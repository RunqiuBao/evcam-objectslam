#ifndef EVENTOBJECTSLAM_H
#define EVENTOBJECTSLAM_H

#include <Eigen/Core>
#include <memory>

namespace eventobjectslam {

typedef Eigen::Matrix4f Mat44_t;
typedef Eigen::Matrix3f Mat33_t;
typedef Eigen::Vector3f Vec3_t;
typedef std::array<float, 3> ObjectExtents;

// -------- double precision, mainly used in g2outils -----------
template<size_t R, size_t C>
using MatRC_d = Eigen::Matrix<double, R, C>;

using Mat33_d = Eigen::Matrix3d;
using Mat44_d = Eigen::Matrix4d;

template<size_t R>
using VecR_d = Eigen::Matrix<double, R, 1>;

using Vec3_d = Eigen::Vector3d;
using Vec6_d = VecR_d<6>;
using Vec7_d = VecR_d<7>;

}

#endif  // EVENTOBJECTSLAM_H