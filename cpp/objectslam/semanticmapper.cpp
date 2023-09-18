#include "semanticmapper.h"

#include <chrono>

#include <logging.h>
TDO_LOGGER("eventobjectslam.semanticmapper")

namespace eventobjectslam {

SemanticMapper::SemanticMapper(std::shared_ptr<MapDataBase> mapDb)
    :_mapDb(mapDb) {}

void SemanticMapper::SchedulePruneLandmarksTask(std::shared_ptr<KeyFrame> pTargetKeyframe) {
    TDO_LOG_DEBUG_FORMAT("scheduled landmark pruning in keyframe(%d)", pTargetKeyframe->_keyFrameID);
    _isPruneLandmarks = true;
    _pTargetKeyframeToPruneLandmark = pTargetKeyframe;
}

void SemanticMapper::_DoPruneLandmarks() {
    TDO_LOG_DEBUG_FORMAT("started landmark pruning in keyframe(%d)", _pTargetKeyframeToPruneLandmark->_keyFrameID);
    auto observedLandmarks_indiceRefObj = _pTargetKeyframeToPruneLandmark->GetObservedLandmarks();
    for (auto pObservedLandmark_indexRefObj : observedLandmarks_indiceRefObj) {
        auto observableKeyframes = _mapDb->GetObservableKeyframes(pObservedLandmark_indexRefObj.first);
        if ((observableKeyframes.size() - pObservedLandmark_indexRefObj.first->GetNumObservations()) > _numNegativeCovisibilityToPruneLandmark) {
            // prune this landmark.
            for (auto pKeyframe_indexRefObj : pObservedLandmark_indexRefObj.first->GetObservations()) {
                pKeyframe_indexRefObj.first->_observedLandmarks_indicesRefObj.erase(pObservedLandmark_indexRefObj.first);
                pKeyframe_indexRefObj.first->_graphNode->UpdateEraseOneCovisibleLandmark(pObservedLandmark_indexRefObj.first->GetObservations());
            }
            _mapDb->_landmarks.erase(pObservedLandmark_indexRefObj.first->_landmarkID);
            TDO_LOG_DEBUG_FORMAT("Erased landmark (%d) from map database and covisibilities of all related keyframes.", pObservedLandmark_indexRefObj.first->_landmarkID);
        }
    }
    // finish task
    _pTargetKeyframeToPruneLandmark = nullptr;
    _isPruneLandmarks = false;
}

void SemanticMapper::Run() {
    TDO_LOG_DEBUG("Start semantic mapper thread.");

    while (!_isTerminate) {
        // if (_isPruneLandmarks) {
        //     auto starttime = std::chrono::steady_clock::now();
        //     _DoPruneLandmarks();
        //     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - starttime);
        //     TDO_LOG_DEBUG_FORMAT("one pruneLandmarks task finished in %d milisec.", duration.count());
        // }
    }
    TDO_LOG_DEBUG("Terminate semantic mapper thread.");
}

}  // end of namespace eventobjectslam