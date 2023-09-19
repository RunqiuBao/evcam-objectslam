#ifndef EVENTOBJECTSLAM_LANDMARK_H
#define EVENTOBJECTSLAM_LANDMARK_H

#include "objectslam.h"
#include "keyframe.h"

#include <mutex>
#include <map>

namespace eventobjectslam {

class KeyFrame;  // Note: due to mutual reference.

class LandMark {

public:
    LandMark(const Mat44_t poseLandmarkInWorld, const std::shared_ptr<object::ObjectBase> pObjectInfo);

    std::map<std::shared_ptr<KeyFrame>, unsigned int> GetObservations() {
        std::lock_guard<std::mutex> lock(_mtxObservations);
        return _observations_indices;
    }

    void AddObservation(std::shared_ptr<KeyFrame> pRefKeyFrame, unsigned int idx);

    // !Note: it is programmer's responsibility to keep the covisibility graph in keyframe side clean.
    void DeleteObservation(std::shared_ptr<KeyFrame> pRefKeyFrame);

    size_t GetNumObservations() { return _observations_indices.size(); }

    float GetDistanceFromBestObserv() { return _distanceFromBestRefKeyframe; }

    Mat44_t GetLandmarkPoseInWorld();
    void SetLandmarkPoseInWorld(const Mat44_t& poseLandmarkInWorld);  // FIXME: could change landmark's bestRefKeyframe

    unsigned int _numObservations = 0;

    const unsigned int _landmarkID;

    const std::shared_ptr<object::ObjectBase> _pObjectInfo;
    Eigen::MatrixXf _vertices3DInLandmark;  // 3D bounding box in landmark frame.

    std::shared_ptr<KeyFrame> _pBestRefKeyFrame;  // Note: closest distances.

    void DeleteThis() { _bIsToDelete = true; }
    bool IsToDelete() { return _bIsToDelete; }

private:
    std::map<std::shared_ptr<KeyFrame>, unsigned int> _observations_indices;  //Note: uint is the refOject index in this keyframe.

    Mat44_t _poseLandmarkInWorld;

    float _bestDetectionScore;  // (deprecated) if detection score becomes better, update detection orientation. 
    float _distanceFromBestRefKeyframe;  // TODO: this is a problamtic design. To deprecate.
    bool _bIsToDelete;

    mutable std::mutex _mtxObservations;
    mutable std::mutex _mtxLandmarkPose;

    static std::atomic<unsigned int> _nextID;
};

}  // end of namespace eventobjectslam

#endif  // EVENTOBJECTSLAM_LANDMARK_H
