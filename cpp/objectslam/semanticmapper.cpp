#include "semanticmapper.h"
#include "optimize/local_bundle_adjust.h"

#include <chrono>

#include <logging.h>
TDO_LOGGER("eventobjectslam.semanticmapper")

namespace eventobjectslam {

SemanticMapper::SemanticMapper(std::shared_ptr<MapDataBase> pMapDb)
    :_pMapDb(pMapDb) {}

void SemanticMapper::SchedulePruneLandmarksTask(std::shared_ptr<KeyFrame> pTargetKeyframe) {
    TDO_LOG_DEBUG_FORMAT("scheduled landmark pruning in keyframe(%d)", pTargetKeyframe->_keyFrameID);
    _isPruneLandmarks = true;
    _pTargetKeyframeToPruneLandmark = pTargetKeyframe;
}

void SemanticMapper::SchedulePruneLandmarksTask() {
    _isPruneLandmarks = true;
}

void SemanticMapper::_DoPruneLandmarks() {
    TDO_LOG_DEBUG_FORMAT("started landmark pruning in keyframe(%d)", _pTargetKeyframeToPruneLandmark->_keyFrameID);
    auto observedLandmarks_indiceRefObj = _pTargetKeyframeToPruneLandmark->GetObservedLandmarks();
    for (auto pObservedLandmark_indexRefObj : observedLandmarks_indiceRefObj) {
        auto observableKeyframes = _pMapDb->GetObservableKeyframes(pObservedLandmark_indexRefObj.first);
        if ((observableKeyframes.size() - pObservedLandmark_indexRefObj.first->GetNumObservations()) > _numNegativeCovisibilityToPruneLandmark) {
            _pMapDb->PruneOneLandmark(pObservedLandmark_indexRefObj.first);
        }
    }
    // finish task
    _pTargetKeyframeToPruneLandmark = nullptr;
    _isPruneLandmarks = false;
}

void SemanticMapper::_DoPruneLandmarks2() {
    TDO_LOG_CRITICAL_FORMAT("------------- NumLandmarks in database before: %d ----------------", _pMapDb->_landmarks.size());
    std::vector<std::shared_ptr<LandMark>> allLandmarksInDb = _pMapDb->GetAllLandmarks();
    for (auto pLandmark : allLandmarksInDb) {
        auto observableKeyframes = _pMapDb->GetObservableKeyframes(pLandmark, _numMinObservableToPruneLandmark);
        TDO_LOG_CRITICAL_FORMAT("landmark(%d): posXYZ %f, %f, %f; numObservs %d; observables %d",
                                    pLandmark->_landmarkID
                                    % pLandmark->GetLandmarkPoseInWorld()(0, 3)
                                    % pLandmark->GetLandmarkPoseInWorld()(1, 3)
                                    % pLandmark->GetLandmarkPoseInWorld()(2, 3)
                                    % pLandmark->GetNumObservations()
                                    % observableKeyframes.size());
        if (pLandmark->GetNumObservations() < _numMinCovisibilityToPruneLandmark && observableKeyframes.size() >= _numMinObservableToPruneLandmark) {
            _pMapDb->PruneOneLandmark(pLandmark);
        }
    }
    TDO_LOG_CRITICAL_FORMAT("------------ NumLandmarks in database after prune: %d --------------", _pMapDb->_landmarks.size());
    _isPruneLandmarks = false;
}

void SemanticMapper::_DoMergeLandmarks() {
    TDO_LOG_DEBUG_FORMAT("NumLandmarks in database before: %d", _pMapDb->_landmarks.size());
    std::vector<std::shared_ptr<LandMark>> allLandmarksInDb = _pMapDb->GetAllLandmarks();
    if (allLandmarksInDb.size() > 1) {
        // TODO: improve this O(N^2) algorithm
        auto smallestAxisObjectExtents = std::min_element(allLandmarksInDb[0]->_pObjectInfo->_objectExtents.begin(), allLandmarksInDb[0]->_pObjectInfo->_objectExtents.end());
        int indexSmallestAxis = std::distance(allLandmarksInDb[0]->_pObjectInfo->_objectExtents.begin(), smallestAxisObjectExtents);
        float distanceThreshold = allLandmarksInDb[0]->_pObjectInfo->_objectExtents[indexSmallestAxis] * 3.;  // Note: 3.0 is a factor.
        std::vector<std::vector<std::shared_ptr<LandMark>>> clusters;
        std::vector<bool> visited(allLandmarksInDb.size(), false);
        for (size_t indexSrcLandmark = 0; indexSrcLandmark < allLandmarksInDb.size(); indexSrcLandmark++) {
            if (!visited[indexSrcLandmark]) {
                std::vector<std::shared_ptr<LandMark>> cluster;
                cluster.push_back(allLandmarksInDb[indexSrcLandmark]);
                visited[indexSrcLandmark] = true;

                for (size_t indexDestLandmark = indexSrcLandmark + 1; indexDestLandmark < allLandmarksInDb.size(); indexDestLandmark++) {
                    if (!visited[indexDestLandmark]) {
                        float distance = (allLandmarksInDb[indexSrcLandmark]->GetLandmarkPoseInWorld().block(0, 3, 3, 1) - allLandmarksInDb[indexDestLandmark]->GetLandmarkPoseInWorld().block(0, 3, 3, 1)).norm();
                        if (distance < distanceThreshold) {
                            cluster.push_back(allLandmarksInDb[indexDestLandmark]);
                            visited[indexDestLandmark] = true;
                        }
                        
                    }
                }

                clusters.push_back(cluster);
            }
        }

        for (auto& cluster : clusters) {
            if (cluster.size() > 1) {
                // merge into one landmark.
                _pMapDb->MergeLandmarkCluster(cluster);
            }
        }
    }
    TDO_LOG_DEBUG_FORMAT("NumLandmarks in database after merge: %d", _pMapDb->_landmarks.size());
    _isPruneLandmarks = false;
}

void SemanticMapper::Run() {
    TDO_LOG_DEBUG("Start semantic mapper thread.");

    while (!_isTerminate) {
        if (_isPruneLandmarks) {
            auto starttime = std::chrono::steady_clock::now();
            _DoPruneLandmarks2();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - starttime);
            TDO_LOG_DEBUG_FORMAT("one pruneLandmarks task finished in %d milisec.", duration.count());
            starttime = std::chrono::steady_clock::now();
            _DoMergeLandmarks();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - starttime);
            TDO_LOG_DEBUG_FORMAT("one mergelandmarks task finished in %d milisec.", duration.count());
        }

        if (_pCurrKeyfrm == nullptr){
            _keyfrmAcceptability = true;
            continue;
        }

        _abortLocalBA = false;
        auto starttime = std::chrono::steady_clock::now();
        optimize::DoLocalBA(_pCurrKeyfrm, &_abortLocalBA);
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - starttime);
        TDO_LOG_DEBUG_FORMAT("one localBA(%d) task finished in %d milisec. keyframe (%d).", _countBA % duration.count() % _pCurrKeyfrm->_keyFrameID);
        _countBA++;

        _keyfrmAcceptability = true;
        _pCurrKeyfrm = nullptr;
    }
    TDO_LOG_CRITICAL("Terminate semantic mapper thread.");
}

bool SemanticMapper::PushKeyframeForBA(std::shared_ptr<KeyFrame> pTargetKeyframe){
    if (_keyfrmAcceptability) {
        _pCurrKeyfrm = pTargetKeyframe;
        _keyfrmAcceptability = false;  // stop receiving keyframes until finish current.
        TDO_LOG_CRITICAL_FORMAT("localBA set for keyframe(%d)", _pCurrKeyfrm->_keyFrameID);
        return true;
    }
    else{
        TDO_LOG_CRITICAL("localBA is busy now.");
        return false;
    }
}

}  // end of namespace eventobjectslam