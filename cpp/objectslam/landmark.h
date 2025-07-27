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
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LandMark(
        const Mat44_t poseLandmarkInWorld,
        const std::vector<Vec3_t> vertices3DInLandmark,
        const float horizontalSize,
        const std::shared_ptr<object::ObjectBase> pObjectInfo,
        const bool hasFacet
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

    float GetScoreFromBestObserv() { return _bestDetectionScore; }

    Mat44_t GetLandmarkPoseInWorld();

    Vec3_t GetOneVertex3DInWorld(size_t indexVertex) const;

    const Eigen::MatrixXf GetVertices3DInLandmark() {
        Eigen::MatrixXf mVertices3D(_vertices3DInLandmark.size(), 3);
        for (int indexVertex=0; indexVertex < _vertices3DInLandmark.size(); indexVertex++){
            mVertices3D.row(indexVertex) = _vertices3DInLandmark[indexVertex];
        }
        return mVertices3D;
    }

    void SetLandmarkPoseInWorld(const Mat44_t& poseLandmarkInWorld);

    void SetVertices3DInLandmark(const std::vector<Vec3_t>& vertices3DInLandmark);

    void SetLandmarkSize(const float observedHeight, const float horizontalSize);

    static void ComputeLandmarkPoseInWorldByVertices3D(
        const std::shared_ptr<KeyFrame> pRefKeyFrame,
        const std::shared_ptr<RefObject> pRefObjInKeyFrame,
        Mat44_t& poseLandmarkInWorld,
        std::vector<Vec3_t>& vertices3DInLandmark
    );

    // ------------------- member variables --------------------------
    unsigned int _numObservations = 0;

    const unsigned int _landmarkID;

    const std::shared_ptr<object::ObjectBase> _pObjectInfo;

    void DeleteThis() { _bIsToDelete = true; }
    bool IsToDelete() { return _bIsToDelete; }

    float _horizontalSize;
    float _observedHeight;
    bool _hasFacet;

private:
    std::map<std::shared_ptr<KeyFrame>, unsigned int> _observations_indices;  //Note: uint is the refOject index in this keyframe.

    Mat44_t _poseLandmarkInWorld;
    std::vector<Vec3_t> _vertices3DInLandmark;

    float _bestDetectionScore;  // if detection score becomes better, update landmark pose and corners.
    bool _bIsToDelete;

    mutable std::mutex _mtxObservations;
    mutable std::mutex _mtxLandmarkPose;
    mutable std::mutex _mtxLandmarkSize;

    static std::atomic<unsigned int> _nextID;
};

}  // end of namespace eventobjectslam

#endif  // EVENTOBJECTSLAM_LANDMARK_H
