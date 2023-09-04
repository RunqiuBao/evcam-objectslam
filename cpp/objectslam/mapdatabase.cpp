#include "mapdatabase.h"

#include <algorithm>
#include <cmath>

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

std::vector<std::shared_ptr<LandMark>> MapDataBase::GetVisibleLandmarks(std::shared_ptr<KeyFrame> pRefKeyFrame){
    float angleFoVLimit = std::atan(std::max(pRefKeyFrame->_pCamera->_cols, pRefKeyFrame->_pCamera->_rows) / 2. /pRefKeyFrame->_pCamera->_kk(0, 0));
    float angleFoVLimitDegree = angleFoVLimit * (180.0 / M_PI);
    TDO_LOG_DEBUG_FORMAT("angleFoVLimit of this camera: %f deg", angleFoVLimitDegree);
    std::vector<std::shared_ptr<LandMark>> visibleLandmarks;
    for (const auto id_landmark : _landmarks){
        Eigen::Vector3f vCamToLandmark = id_landmark.second->_poseLandmarkInWorld.block(0, 3, 3, 1) - pRefKeyFrame->_poseCurrentFrameInWorld.block(0, 3, 3, 1);
        vCamToLandmark /= vCamToLandmark.norm();
        Eigen::Vector3f vCamZ = pRefKeyFrame->_poseCurrentFrameInWorld.block(0, 2, 3, 1);
        float angleViewRay = std::acos(vCamToLandmark.dot(vCamZ));
        if (angleViewRay < angleFoVLimit){
            visibleLandmarks.push_back(id_landmark.second);
        }
    }
    TDO_LOG_DEBUG_FORMAT("Found %d visible landmarks in total %d landmarks.", visibleLandmarks.size() % _landmarks.size());
    return visibleLandmarks;
}

} // end of namespace eventobjectslam