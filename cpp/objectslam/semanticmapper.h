#ifndef EVENTOBJECTSLAM_SEMANTICMAPPER_H
#define EVENTOBJECTSLAM_SEMANTICMAPPER_H

#include "mapdatabase.h"

namespace eventobjectslam {

class SemanticMapper {

public:
    // Constructor
    SemanticMapper(std::shared_ptr<MapDataBase> mapDb);

    // Destructor
    ~SemanticMapper() = default;

    // main loop
    void Run();

    void Terminate() { _isTerminate = true; }
    void SchedulePruneLandmarksTask(std::shared_ptr<KeyFrame> pTargetKeyframe);
    void SchedulePruneLandmarksTask();
    bool PushKeyframeForBA(std::shared_ptr<KeyFrame> pTargetKeyframe);

    static constexpr size_t _numNegativeCovisibilityToPruneLandmark = 5;  //!Note: if more than this of covisibility can not see the landmark, then prune it.
    static constexpr size_t _numMinCovisibilityToPruneLandmark = 2;
    static constexpr size_t _numMinObservableToPruneLandmark = 10;

private:
    void _DoPruneLandmarks();  // every new keyframes, scan once the 5th newest keyframe and prune observed landmarks that are not observed enough overall.
    void _DoPruneLandmarks2();  // every Nth keyframes created, scan once if there are landmarks observed less than _numMinCovisibilityToPruneLandmark times, but could be observed _numMinObservableToPruneLandmark times.
    /*
     *  Assume all landmarks are same type. Merge them if they are within certain physical distance.
     */
    void _DoMergeLandmarks();  // every Nth keyframes created, scan once if there are landmarks suspiciouly close to each other. If so, then merge.

    const std::shared_ptr<MapDataBase> _pMapDb;

    bool _isTerminate = false;
    bool _isPruneLandmarks = false;
    std::shared_ptr<KeyFrame> _pTargetKeyframeToPruneLandmark = nullptr;

    //flag to abort local BA
    bool _abortLocalBA = false;

    // //! queue for keyframes
    // std::list<std::shared_ptr<KeyFrame>> _keyfrmsQueue;

    // //! mutex for access to keyframe queue
    // mutable std::mutex _mtxKeyfrmQueue;

    //! current keyframe which is used in the current mapping
    std::shared_ptr<KeyFrame> _currKeyfrm = nullptr;

    //! flag for keyframe acceptability
    std::atomic<bool> _keyfrmAcceptability{true};

};

}  // end of namespace eventobjectslam


#endif