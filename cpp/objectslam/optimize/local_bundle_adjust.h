#ifndef EVENTOBJECTSLAM_OPTIMIZE_LOCALBA_H
#define EVENTOBJECTSLAM_OPTIMIZE_LOCALBA_H

#include "keyframe.h"

namespace eventobjectslam {

namespace optimize {

void DoLocalBA(std::shared_ptr<KeyFrame> pCurrKeyframe, bool* const bForceStopFlag, const size_t numFirstIter = 5, const size_t numSecondIter = 10, const float maxPoseErrorBA = 0.2f);

}  // end of namspace optimize


}  // end of namespace eventobjectslam

#endif