#include "mapdatabase.h"
#include "mathutils.h"

#include <algorithm>
#include <cmath>
#include <chrono>

#include <logging.h>
TDO_LOGGER("eventobjectslam.mapdatabase")


namespace eventobjectslam{

std::mutex MapDataBase::_mtxDatabase;

MapDataBase::MapDataBase(){
    TDO_LOG_INFO("Construct: MapDataBase");
}

void MapDataBase::AddKeyFrame(std::shared_ptr<KeyFrame> pKeyFrame){
    std::lock_guard<std::mutex> lock(_mtxMapKeyframesAccess);
    _keyframes[pKeyFrame->_keyFrameID] = pKeyFrame;
    if (pKeyFrame->_keyFrameID > _maxKeyFrameID)
        _maxKeyFrameID = pKeyFrame->_keyFrameID;
}

void MapDataBase::AddLandMark(std::shared_ptr<LandMark> pLandmark){
    std::lock_guard<std::mutex> lock(_mtxMapLandmarksAccess);
    _landmarks[pLandmark->_landmarkID] = pLandmark;
    TDO_LOG_INFO_FORMAT("Added one landmark to MapDB. Totally %d landmarks.", _landmarks.size());
}

std::vector<std::shared_ptr<LandMark>> MapDataBase::GetVisibleLandmarks(std::shared_ptr<KeyFrame> pRefKeyFrame) {
    std::lock_guard<std::mutex> lock(_mtxMapLandmarksAccess);
    float angleFoVLimit = std::atan(std::max(pRefKeyFrame->_pCamera->_cols, pRefKeyFrame->_pCamera->_rows) / 2. /pRefKeyFrame->_pCamera->_kk(0, 0));
    float angleFoVLimitDegree = angleFoVLimit * (180.0 / M_PI);
    TDO_LOG_DEBUG_FORMAT("angleFoVLimit of this camera: %f deg", angleFoVLimitDegree);
    std::vector<std::shared_ptr<LandMark>> visibleLandmarks;
    for (const auto id_landmark : _landmarks){
        Eigen::Vector3f vCamToLandmark = id_landmark.second->GetLandmarkPoseInWorld().block(0, 3, 3, 1) - pRefKeyFrame->GetKeyframePoseInWorld().block(0, 3, 3, 1);
        vCamToLandmark /= vCamToLandmark.norm();
        Eigen::Vector3f vCamZ = pRefKeyFrame->GetKeyframePoseInWorld().block(0, 2, 3, 1);
        float angleViewRay = std::acos(vCamToLandmark.dot(vCamZ));
        if (angleViewRay < angleFoVLimit){
            visibleLandmarks.push_back(id_landmark.second);
        }
    }
    TDO_LOG_DEBUG_FORMAT("Found %d visible landmarks in total %d landmarks.", visibleLandmarks.size() % _landmarks.size());
    return visibleLandmarks;
}

std::vector<std::shared_ptr<KeyFrame>> MapDataBase::GetObservableKeyframes(std::shared_ptr<LandMark> pRefLandmark, size_t numKeyfrmToStop) {
    std::lock_guard<std::mutex> lock(_mtxMapKeyframesAccess);
    auto starttime = std::chrono::steady_clock::now();
    std::vector<std::shared_ptr<KeyFrame>> observableKeyframes;
    for (const auto id_keyframe : _keyframes) {
        float angleFoVLimit = std::atan(std::max(id_keyframe.second->_pCamera->_cols, id_keyframe.second->_pCamera->_rows) / 2. / id_keyframe.second->_pCamera->_kk(0, 0));
        float angleFoVLimitDegree = angleFoVLimit * (180.0 / M_PI);
        TDO_LOG_VERBOSE_FORMAT("angleFoVLimit of this camera: %f deg", angleFoVLimitDegree);
        Eigen::Vector3f vCamToLandmark =  pRefLandmark->GetLandmarkPoseInWorld().block(0, 3, 3, 1) - id_keyframe.second->GetKeyframePoseInWorld().block(0, 3, 3, 1);
        vCamToLandmark /= vCamToLandmark.norm();
        Eigen::Vector3f vCamZ = id_keyframe.second->GetKeyframePoseInWorld().block(0, 2, 3, 1);
        float angleViewRay = std::acos(vCamToLandmark.dot(vCamZ));
        TDO_LOG_VERBOSE_FORMAT("angleViewRay: %f, angleFoVLimit: %f", angleViewRay % angleFoVLimit);
        if (angleViewRay < angleFoVLimit){
            // check if the projection is within image and large enough
            Eigen::MatrixXf transformedVerticesInWorld = mathutils::TransformPoints<Eigen::MatrixXf>(pRefLandmark->GetLandmarkPoseInWorld(), pRefLandmark->_vertices3DInLandmark);
            Eigen::MatrixXf transformedVerticesInCamera = mathutils::TransformPoints<Eigen::MatrixXf>((id_keyframe.second->GetKeyframePoseInWorld()).inverse(), transformedVerticesInWorld);
            std::vector<cv::Point> oneLandmarkPoints2D = mathutils::ProjectPoints3DToPoints2D(transformedVerticesInCamera, *(id_keyframe.second->_pCamera));
            const size_t cameraCols = id_keyframe.second->_pCamera->_cols;
            const size_t cameraRows = id_keyframe.second->_pCamera->_rows;
            float minX = cameraCols;
            float maxX = 0;
            float minY = cameraRows;
            float maxY = 0;
            bool bIsObservable = true;
            for (cv::Point onePoint2D : oneLandmarkPoints2D) {
                if (
                    onePoint2D.x < 0
                    || onePoint2D.x > cameraCols
                    || onePoint2D.y < 0
                    || onePoint2D.y > cameraRows
                ){
                    TDO_LOG_VERBOSE("skip due to onePoint2D: " << std::to_string(onePoint2D.x) << ", " << std::to_string(onePoint2D.y));
                    bIsObservable = false;
                    break;  // not completely observable.
                }
                maxX = onePoint2D.x > maxX ? onePoint2D.x : maxX;
                minX = onePoint2D.x < minX ? onePoint2D.x : minX;
                maxY = onePoint2D.y > maxY ? onePoint2D.y : maxY;
                minY = onePoint2D.y < minY ? onePoint2D.y : minY;
            }
            float projectionArea = (maxX - minX) * (maxY - minY);
            TDO_LOG_VERBOSE_FORMAT("projection area: %f", projectionArea);
            if (bIsObservable && projectionArea > _minAreaAsGoodObservation) {
                observableKeyframes.push_back(id_keyframe.second);
            }
            if (numKeyfrmToStop > 0 && observableKeyframes.size() >= numKeyfrmToStop) {
                break;
            }
        }
    }
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - starttime);
    TDO_LOG_DEBUG_FORMAT("scan observable keyframes for landmark (%d) took %d milisec.", pRefLandmark->_landmarkID % duration.count());
    return observableKeyframes;
}

std::vector<std::shared_ptr<KeyFrame>> MapDataBase::GetAllKeyframes() const{
    std::lock_guard<std::mutex> lock(_mtxMapKeyframesAccess);
    std::vector<std::shared_ptr<KeyFrame>> keyframes;
    keyframes.reserve(_keyframes.size());
    for (const auto id_keyframe : _keyframes) {
        keyframes.push_back(id_keyframe.second);
    }
    return keyframes;
}

std::vector<std::shared_ptr<LandMark>> MapDataBase::GetAllLandmarks() const{
    std::lock_guard<std::mutex> lock(_mtxMapLandmarksAccess);
    std::vector<std::shared_ptr<LandMark>> landmarks;
    landmarks.reserve(_landmarks.size());
    for (const auto id_landmark : _landmarks) {
        landmarks.push_back(id_landmark.second);
    }
    return landmarks;
}

void MapDataBase::PruneOneLandmark(std::shared_ptr<LandMark> oneLandmarkToPrune) {
    std::lock_guard<std::mutex> lock(_mtxMapLandmarksAccess);
    oneLandmarkToPrune->DeleteThis();
    // prune this landmark.
    for (auto pKeyframe_indexRefObj : oneLandmarkToPrune->GetObservations()) {
        pKeyframe_indexRefObj.first->DeleteOneObservedLandmark(oneLandmarkToPrune);
    }
    _landmarks.erase(oneLandmarkToPrune->_landmarkID);
    TDO_LOG_DEBUG_FORMAT("Erased landmark (%d) from map database and covisibilities of all related keyframes.", oneLandmarkToPrune->_landmarkID);
}

void MapDataBase::MergeLandmarkCluster(std::vector<std::shared_ptr<LandMark>> oneCluster) {
    std::lock_guard<std::mutex> lock(_mtxMapLandmarksAccess);
    size_t indexBestLandmark = 0;
    for (size_t indexLandmark = 1; indexLandmark < oneCluster.size(); indexLandmark++) {
        if (oneCluster[indexLandmark]->GetDistanceFromBestObserv() < oneCluster[indexBestLandmark]->GetDistanceFromBestObserv()) {
            indexBestLandmark = indexLandmark;
        }
    }
    for (size_t indexLandmark = 0; indexLandmark < oneCluster.size(); indexLandmark++) {
        if (indexLandmark != indexBestLandmark) {
            for (auto pKeyframe_indexRefObj : oneCluster[indexLandmark]->GetObservations()) {
                oneCluster[indexLandmark]->DeleteThis();
                pKeyframe_indexRefObj.first->ReplaceOneObservedLandmark(oneCluster[indexLandmark], oneCluster[indexBestLandmark]);
            }
            _landmarks.erase(oneCluster[indexLandmark]->_landmarkID);
        }
    }

}

} // end of namespace eventobjectslam
