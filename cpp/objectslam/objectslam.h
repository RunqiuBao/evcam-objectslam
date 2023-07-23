#ifndef EVENTOBJECTSLAM_H
#define EVENTOBJECTSLAM_H

#include <Eigen/Core>
#include <memory>

namespace eventobjectslam {

typedef Eigen::Matrix4f Mat44_t;
typedef std::array<float, 3> ObjectExtents;

}

#endif  // EVENTOBJECTSLAM_H