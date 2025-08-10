#include "semanticmapper.h"
#include "optimize/local_bundle_adjust.h"
#include "keyframe.h"
#include "mathutils.h"

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

void SemanticMapper::_DoPruneKeyframes(std::shared_ptr<KeyFrame> pCurrKeyfrm, const size_t maxNumKeyfrm) {
    float ratioToPrune = 0.5;  // inline parameter
    TDO_LOG_CRITICAL_FORMAT("NumKeyframes in database before: %d", _pMapDb->GetAllKeyframes().size());
    // when a new keyframe becomes the best keyframe in a subset of covisibility. Those other keyframes in this subset (fullCovisibleSubset) can be pruned.
    std::vector<std::shared_ptr<KeyFrame>> vFullCovisibilities = pCurrKeyfrm->GetOrderedFullCovisibilities();
    size_t numToPrune = std::min(_pMapDb->GetAllKeyframes().size() - maxNumKeyfrm, static_cast<size_t>(vFullCovisibilities.size() * ratioToPrune));
    std::vector<size_t>  indiciesToPrune = mathutils::GetListOfRandomIndex(0, vFullCovisibilities.size(), numToPrune);
    std::vector<std::shared_ptr<KeyFrame>> keyframesToPrune(indiciesToPrune.size()); 
    for (size_t indexToPrune : indiciesToPrune) {
        if (!vFullCovisibilities[indexToPrune]){
            TDO_LOG_CRITICAL_FORMAT("get one empty pKeyfram in covisibilities. (%d)", vFullCovisibilities.size());
            continue;
        }
        keyframesToPrune.push_back(vFullCovisibilities[indexToPrune]);
    }
    for (auto pKeyframe : keyframesToPrune) {
        if (!pKeyframe) {
            continue;  // empty pointer
        }
        if (pKeyframe->_keyFrameID == 0){  // Note: do not delete the first keyframe.
            continue;
        }
        if (pKeyframe->_keyFrameID >= pCurrKeyfrm->_keyFrameID) {
            TDO_LOG_DEBUG_FORMAT("do not prune covisibility keyframes (%d) that are newer than currKeyframe (%d).", pKeyframe->_keyFrameID % pCurrKeyfrm->_keyFrameID);
            continue;  // Note: do not delete the current keyframe.
        }
        TDO_LOG_CRITICAL_FORMAT("pruning pKeyframe (%d), _pCurrKeyfrm is (%d)", pKeyframe->_keyFrameID % _pCurrKeyfrm->_keyFrameID);
        _pMapDb->PruneOneKeyframe(pKeyframe);
    }
    TDO_LOG_CRITICAL_FORMAT("NumKeyframes in database after: %d", _pMapDb->GetAllKeyframes().size());
}

void SemanticMapper::_DoPruneLandmarks2() {
    TDO_LOG_DEBUG_FORMAT("------------- NumLandmarks in database before: %d ----------------", _pMapDb->_landmarks.size());
    std::vector<std::shared_ptr<LandMark>> allLandmarksInDb = _pMapDb->GetAllLandmarks();
    
    for (auto pLandmark : allLandmarksInDb) {
        auto observableKeyframes = _pMapDb->GetObservableKeyframes(pLandmark, _numMinObservableToPruneLandmark);
        TDO_LOG_DEBUG_FORMAT("landmark(%d): posXYZ %f, %f, %f; numObservs %d; observables %d",
                                    pLandmark->_landmarkID
                                    % pLandmark->GetLandmarkPoseInWorld()(0, 3)
                                    % pLandmark->GetLandmarkPoseInWorld()(1, 3)
                                    % pLandmark->GetLandmarkPoseInWorld()(2, 3)
                                    % pLandmark->GetNumObservations()
                                    % observableKeyframes.size());
        if (
            pLandmark->GetNumObservations() < _numMinCovisibilityToPruneLandmark && observableKeyframes.size() >= _numMinObservableToPruneLandmark  // more than _numMinObservableToPruneLandmark of keyframes can see it, but only _numMinCovisibilityToPruneLandmark saw it.
            || pLandmark->GetNumObservations() == 0  // no keyframe saw it.
        ) {
            _pMapDb->PruneOneLandmark(pLandmark);
        }
    }
    TDO_LOG_DEBUG_FORMAT("------------ NumLandmarks in database after prune: %d --------------", _pMapDb->_landmarks.size());
    _isPruneLandmarks = false;
}

void SemanticMapper::_DoMergeLandmarks() {
    TDO_LOG_DEBUG_FORMAT("NumLandmarks in database before: %d", _pMapDb->_landmarks.size());
    std::vector<std::shared_ptr<LandMark>> allLandmarksInDb = _pMapDb->GetAllLandmarks();
    if (allLandmarksInDb.size() > 1) {
        // TODO: improve this O(N^2) algorithm
        std::vector<std::vector<std::shared_ptr<LandMark>>> clusters;
        std::vector<bool> visited(allLandmarksInDb.size(), false);
        for (size_t indexSrcLandmark = 0; indexSrcLandmark < allLandmarksInDb.size(); indexSrcLandmark++) {
            if (!visited[indexSrcLandmark]) {
                std::vector<std::shared_ptr<LandMark>> cluster;
                cluster.push_back(allLandmarksInDb[indexSrcLandmark]);
                visited[indexSrcLandmark] = true;

                for (size_t indexDestLandmark = indexSrcLandmark + 1; indexDestLandmark < allLandmarksInDb.size(); indexDestLandmark++) {
                    if (!visited[indexDestLandmark]) {
                        float distanceThreshold = allLandmarksInDb[indexDestLandmark]->_horizontalSize * 3.;  // Note: 3.0 is a factor.
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
    size_t maxNumKeyframesInFullCovisibleSubset = 2000;  // inline parameter
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

        if (_pMapDb->GetAllKeyframes().size() > maxNumKeyframesInFullCovisibleSubset){
            starttime = std::chrono::steady_clock::now();
            _DoPruneKeyframes(_pCurrKeyfrm, maxNumKeyframesInFullCovisibleSubset);
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - starttime);
            TDO_LOG_DEBUG_FORMAT("one pruneKeyframes task finished in %d milisec.", duration.count());
        }

        _keyfrmAcceptability = true;
        _pCurrKeyfrm = nullptr;
    }
    TDO_LOG_DEBUG("Terminate semantic mapper thread.");
}

bool SemanticMapper::PushKeyframeForBA(std::shared_ptr<KeyFrame> pTargetKeyframe){
    if (_keyfrmAcceptability) {
        _pCurrKeyfrm = pTargetKeyframe;
        _keyfrmAcceptability = false;  // stop receiving keyframes until finish current.
        TDO_LOG_DEBUG_FORMAT("localBA set for keyframe(%d)", _pCurrKeyfrm->_keyFrameID);
        return true;
    }
    else{
        TDO_LOG_DEBUG("localBA is busy now.");
        return false;
    }
}

}  // end of namespace eventobjectslam