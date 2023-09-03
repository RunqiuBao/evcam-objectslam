#include "mapdatabase.h"

#include <logging.h>
TDO_LOGGER("eventobjectslam.mapdatabase")


namespace eventobjectslam{

MapDataBase::MapDataBase(){
    TDO_LOG_INFO("Construct: MapDataBase");
}

void MapDataBase::AddKeyFrame(std::shared_ptr<KeyFrame> pKeyFrame){
    std::lock_guard<std::mutex> lock(_mtxMapAccess);
    _keyFrames[pKeyFrame->_keyFrameID] = pKeyFrame;
    if (pKeyFrame->_keyFrameID > _maxKeyFrameID)
        _maxKeyFrameID = pKeyFrame->_keyFrameID;
}

void MapDataBase::AddLandMark(std::shared_ptr<LandMark> pLandmark){
    std::lock_guard<std::mutex> lock(_mtxMapAccess);
    _landmarks[pLandmark->_landmarkID] = pLandmark;
}

} // end of namespace eventobjectslam