#ifndef EVENTOBJECTSLAM_MAPDATABASE_H
#define EVENTOBJECTSLAM_MAPDATABASE_H

#include "keyframe.h"
#include "landmark.h"

#include <opencv2/core.hpp>

#include <mutex>
#include <unordered_map>

namespace eventobjectslam{


/*
 *  (1) hold all the keyframes
 *  (2) hold all the landmarks
 *  (3) publish keyframes and landmarks to pangolin viewer
 *
 */
class MapDataBase {

public:
    MapDataBase();

    // ~MapDataBase();

    void AddKeyFrame(std::shared_ptr<KeyFrame> pKeyFrame);

    void AddLandMark(std::shared_ptr<LandMark> pLandmark);

    // scan existing landmarks for ones that are within fov of input keyFrame.
    std::vector<std::shared_ptr<LandMark>> GetVisibleLandmarks(std::shared_ptr<KeyFrame> pRefKeyFrame);

    // scan existing keyframes for ones that can see the landmark with good resolution.
    std::vector<std::shared_ptr<KeyFrame>> GetObservableKeyframes(std::shared_ptr<LandMark> pRefLandmark, size_t numKeyfrmToStop = 0);
    static constexpr size_t _minAreaAsGoodObservation = 666;  //!Note: unit is pixels

    std::vector<std::shared_ptr<KeyFrame>> GetAllKeyframes() const;

    void PruneOneKeyframe(std::shared_ptr<KeyFrame> pOneKeyframeToPrune);

    std::vector<std::shared_ptr<LandMark>> GetAllLandmarks() const;

    void PruneOneLandmark(std::shared_ptr<LandMark> oneLandmarkToPrune);

    void MergeLandmarkCluster(std::vector<std::shared_ptr<LandMark>> oneCluster);

    size_t _maxKeyFrameID = 0;
    //! IDs and keyframes
    std::unordered_map<unsigned int, std::shared_ptr<KeyFrame>> _keyframes;

    //! IDs and landmarks
    std::unordered_map<unsigned int, std::shared_ptr<LandMark>> _landmarks;

    //! mutex for locking ALL access to the database
    //!NOTE: cannot be used in map_database class. Only use from outside.
    static std::mutex _mtxDatabase;

    void SetObjectSize(const Vec3_t& objectSize) {_objectSize = objectSize;}
    const Vec3_t GetObjectSize(){return _objectSize;}

    // publish the per-frame debug view image (stereo frame with detections and tracking status) to the viewer thread
    void SetDebugViewImage(const cv::Mat& debugViewImage) {
        std::lock_guard<std::mutex> lock(_mtxDebugViewAccess);
        _debugViewImage = debugViewImage;
    }
    cv::Mat GetDebugViewImage() const {
        std::lock_guard<std::mutex> lock(_mtxDebugViewAccess);
        return _debugViewImage;
    }

private:
    // mutex for mutal exclusion controll between class methods called in different threads
    mutable std::mutex _mtxMapKeyframesAccess;

    mutable std::mutex _mtxMapLandmarksAccess;

    mutable std::mutex _mtxDebugViewAccess;

    cv::Mat _debugViewImage;

    Vec3_t _objectSize;

};

}  // end of namespace eventobjectslam

#endif