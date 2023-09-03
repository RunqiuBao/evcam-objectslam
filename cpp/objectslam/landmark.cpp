#include "landmark.h"

#include <logging.h>
TDO_LOGGER("eventobjectslam.landmark")

namespace eventobjectslam {

std::atomic<unsigned int> LandMark::_nextID{0};

LandMark::LandMark(const Mat44_t pose_wc)
:_pose_wc(pose_wc), _landmarkID(_nextID++){}

}  // end of namespace eventobjectslam