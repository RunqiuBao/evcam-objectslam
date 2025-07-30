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
            Eigen::MatrixXf transformedVerticesInWorld = mathutils::TransformPoints<Eigen::MatrixXf>(pRefLandmark->GetLandmarkPoseInWorld(), pRefLandmark->GetVertices3DInLandmark());
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

void MapDataBase::PruneOneKeyframe(std::shared_ptr<KeyFrame> pOneKeyframeToPrune) {
    size_t rangeToSearchNeighbor = 5;  // inline parameter
    float maxDistanceToPrune = 0.2;  // inline parameter
    if (pOneKeyframeToPrune->_bShouldNotPrune) {
        TDO_LOG_CRITICAL("once a good neighbor, skip. pruneKeyframe failed");
        return;
    }
    pOneKeyframeToPrune->DeleteThis();
    // prune this keyframe from refered frames, landmarks, covisibilities of other keyframes, map database.
    {
        std::lock_guard<std::mutex> lock(_mtxMapKeyframesAccess);
        _keyframes.erase(pOneKeyframeToPrune->_keyFrameID);
    }
    auto covisibilities = pOneKeyframeToPrune->GetOrderedCovisibilities();
    if (covisibilities.size() < rangeToSearchNeighbor) {
        TDO_LOG_CRITICAL_FORMAT("not enough covisibilities. pruneKeyframe failed. (numCovisible: %d / %d)", covisibilities.size() % rangeToSearchNeighbor);
        return;
    }
    size_t indexClosest = 0;
    std::shared_ptr<KeyFrame> pKeyframeNeighbor; // the landmarks and frames in pOneKeyframeToPrune will be moved to stay under this pKeyframeNeighbor.
    float distance = std::numeric_limits<float>::max();
    for (size_t indexCovisible = 0; indexCovisible < rangeToSearchNeighbor; indexCovisible++) {
        float newDistance = (covisibilities[indexCovisible]->GetKeyframePoseInWorld().block(0, 3, 3, 1) - pOneKeyframeToPrune->GetKeyframePoseInWorld().block(0, 3, 3, 1)).norm();
        if (newDistance < distance) {
            distance = newDistance;
            pKeyframeNeighbor = covisibilities[indexCovisible];
        }
    }
    // check if the neighbor is full covisible to the one to prune and within a physical range. If not, stop pruning.
    auto profileNeighbor = pKeyframeNeighbor->GetProfileOfObservedLandmarks();
    auto profileKfmToPrune = pOneKeyframeToPrune->GetProfileOfObservedLandmarks();
    if (!std::includes(profileNeighbor.begin(), profileNeighbor.end(), profileKfmToPrune.begin(), profileKfmToPrune.end())  // best neighbor is not covisible to the one to prune
        || distance >= maxDistanceToPrune  // best neighbor is too far away from the one to prune
    ) {
        TDO_LOG_CRITICAL_FORMAT("can't find a good neighbor. pruneKeyframe failed. (distance: %f / %f)", distance % maxDistanceToPrune);
        return;  // prune failed.
    }

    /* start pruning */
    pKeyframeNeighbor->_bShouldNotPrune = true;  // hack to keep keyframes distributed.
    for (auto pFrame_id : pOneKeyframeToPrune->_vFrames_ids) {
        Mat44_t pose_nk = pKeyframeNeighbor->GetKeyframePoseInWorld().inverse() * pOneKeyframeToPrune->GetKeyframePoseInWorld();
        Mat44_t pose_nc = pose_nk * pFrame_id.first->GetPose();
        pFrame_id.first->SetPose(pose_nc);
        pFrame_id.first->_pRefKeyframe = pKeyframeNeighbor;
        if (!pKeyframeNeighbor->_vFrames_ids.count(pFrame_id.first)){
            pKeyframeNeighbor->_vFrames_ids[pFrame_id.first] = pFrame_id.second;
        }
    }

    for(auto pLandmark_indexRefObj : pOneKeyframeToPrune->GetObservedLandmarks()) {
        pLandmark_indexRefObj.first->DeleteObservation(pOneKeyframeToPrune);
    }

    for (auto pKeyframe : covisibilities) {
        if (!pKeyframe->IsToDelete()) {
            pKeyframe->DeleteCovisibilityConnection(pOneKeyframeToPrune);
        }
    }
    TDO_LOG_CRITICAL_FORMAT("Erased keyframe (%d) from refered frames, landmarks, covisibilities of other keyframes, map database", pOneKeyframeToPrune->_keyFrameID);
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
        if (oneCluster[indexLandmark]->GetScoreFromBestObserv() > oneCluster[indexBestLandmark]->GetScoreFromBestObserv()) {
            indexBestLandmark = indexLandmark;
        }
    }
    for (size_t indexLandmark = 0; indexLandmark < oneCluster.size(); indexLandmark++) {
        if (indexLandmark != indexBestLandmark) {
            for (auto pKeyframe_indexRefObj : oneCluster[indexLandmark]->GetObservations()) {
                oneCluster[indexBestLandmark]->AddObservation(pKeyframe_indexRefObj.first, pKeyframe_indexRefObj.second);
                oneCluster[indexLandmark]->DeleteThis();
                pKeyframe_indexRefObj.first->ReplaceOneObservedLandmark(oneCluster[indexLandmark], oneCluster[indexBestLandmark]);
            }
            _landmarks.erase(oneCluster[indexLandmark]->_landmarkID);
        }
    }

}

} // end of namespace eventobjectslam
