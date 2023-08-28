#ifndef EVENTOBJECTSLAM_LANDMARK_H
#define EVENTOBJECTSLAM_LANDMARK_H

#include "objectslam.h"

namespace eventobjectslam {

class LandMark {

public:
    LandMark(const Vec3_t posInW);

    unsigned int _numObservations = 0;

private:
    std::map<std::shared_ptr<KeyFrame>, unsigned int> _observations;

    mutable std::mutex _mtxObservations;
}

}  // end of namespace eventobjectslam

#endif  // EVENTOBJECTSLAM_LANDMARK_H