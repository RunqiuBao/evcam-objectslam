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
    LandMark(
        const Mat44_t poseLandmarkInWorld,
        const Vec3_t keypt1InLandmark,
        const float horizontalSize,
        const std::shared_ptr<object::ObjectBase> pObjectInfo
    );

    std::map<std::shared_ptr<KeyFrame>, unsigned int> GetObservations() {
        std::lock_guard<std::mutex> lock(_mtxObservations);
        return _observations_indices;
    }

    void AddObservation(std::shared_ptr<KeyFrame> pRefKeyFrame, unsigned int idx);

    bool CheckIfObservation(std::shared_ptr<KeyFrame> pRefKeyFrame);

    // !Note: it is programmer's responsibility to keep the covisibility graph in keyframe side clean.
    void DeleteObservation(std::shared_ptr<KeyFrame> pRefKeyFrame);

    size_t GetNumObservations() { return _observations_indices.size(); }

    float GetDistanceFromBestObserv() { return _distanceFromBestRefKeyframe; }

    Mat44_t GetLandmarkPoseInWorld();

    Vec3_t GetKeypt1InLandmark() const;

    void SetLandmarkPoseInWorld(const Mat44_t& poseLandmarkInWorld);

    void SetKeypt1InLandmark(const Vec3_t& keypt1InLandmark);

    void SetLandmarkSize(const float observedHeight, const float horizontalSize);

    static void ComputeLandmarkPoseInWorldAndKeypt1InWolrd(
        const std::shared_ptr<KeyFrame> pRefKeyFrame,
        const std::shared_ptr<RefObject> pRefObjInKeyFrame,
        Mat44_t& poseLandmarkInWorld,
        Vec3_t& keypt1InLandmark
    );

    // ------------------- member variables --------------------------
    unsigned int _numObservations = 0;

    const unsigned int _landmarkID;

    const std::shared_ptr<object::ObjectBase> _pObjectInfo;
    Eigen::MatrixXf _vertices3DInLandmark;  // 3D bounding box in landmark frame.
    float _horizontalSize;  // diameter of a cylinder.
    float _observedHeight;  // height of the cylinder = 2 * norm(oc - keypt1)

    void DeleteThis() { _bIsToDelete = true; }
    bool IsToDelete() { return _bIsToDelete; }

private:
    std::map<std::shared_ptr<KeyFrame>, unsigned int> _observations_indices;  //Note: uint is the refOject index in this keyframe.

    Mat44_t _poseLandmarkInWorld;
    Vec3_t _keypt1InLandmark;

    float _bestDetectionScore;  // (deprecated) if detection score becomes better, update detection orientation. 
    float _distanceFromBestRefKeyframe;  // TODO: this is a problamtic design. To deprecate.
    bool _bIsToDelete;

    mutable std::mutex _mtxObservations;
    mutable std::mutex _mtxLandmarkPose;
    mutable std::mutex _mtxLandmarkSize;

    static std::atomic<unsigned int> _nextID;
};

}  // end of namespace eventobjectslam

#endif  // EVENTOBJECTSLAM_LANDMARK_H
