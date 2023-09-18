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

private:
    void _DoPruneLandmarks();

    const std::shared_ptr<MapDataBase> _mapDb;

    bool _isTerminate = false;
    bool _isPruneLandmarks = false;
    std::shared_ptr<KeyFrame> _pTargetKeyframeToPruneLandmark = nullptr;
    static constexpr size_t _numNegativeCovisibilityToPruneLandmark = 5;  //!Note: if more than this of covisibility can not see the landmark, then prune it.
};

}  // end of namespace eventobjectslam


#endif