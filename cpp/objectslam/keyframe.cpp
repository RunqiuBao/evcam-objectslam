#include "keyframe.h"

#include <logging.h>
TDO_LOGGER("eventobjectslam.keyframe")

namespace eventobjectslam {

KeyFrame::KeyFrame(const std::shared_ptr<Frame> pRefFrame, const Mat44_t& refKeyFrameInWorldTransform){
    _pose_wc = refKeyFrameInWorldTransform * pRefFrame->GetPose();
    size_t countLandmark = 0;
    for(ThreeDDetection oneDetection : pRefFrame->_threeDDetections){
        std::shared_ptr<LandMark> oneLandmark = std::make_shared<LandMark>(oneDetection, countLandmark);
        landmarks.push_back(oneLandmark);
        countLandmark++;
    }
    _pRefFrame = pRefFrame;
}

} // end of namespace eventobjectslam
