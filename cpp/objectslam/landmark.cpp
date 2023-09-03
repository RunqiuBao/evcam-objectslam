#include "landmark.h"

#include <logging.h>
TDO_LOGGER("eventobjectslam.landmark")

namespace eventobjectslam {

std::atomic<unsigned int> LandMark::_nextID{0};

LandMark::LandMark(const Mat44_t poseCurrentFrameInWorld, const std::shared_ptr<object::ObjectBase> pObjectInfo)
:_poseCurrentFrameInWorld(poseCurrentFrameInWorld), _landmarkID(_nextID++), _pObjectInfo(pObjectInfo){

}

}  // end of namespace eventobjectslam