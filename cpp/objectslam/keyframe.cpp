#include "keyframe.h"

#include <logging.h>
TDO_LOGGER("eventobjectslam.keyframe")

namespace eventobjectslam {

KeyFrame::KeyFrame(const Frame& refFrame){
    _pose_wc = refFrame.GetPose();
    size_t countLandmark = 0;
    for(ThreeDDetection oneDetection : refFrame._threeDDetections){
        std::shared_ptr<LandMark> oneLandmark = std::make_shared<LandMark>(oneDetection, countLandmark);
        landmarks.push_back(oneLandmark);
        countLandmark++;
    }
}

} // end of namespace eventobjectslam
