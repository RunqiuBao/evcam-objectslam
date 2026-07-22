#ifndef EVENTOBJECTSLAM_SYSTEM_H
#define EVENTOBJECTSLAM_SYSTEM_H

#include "objectslam.h"
#include "frametracker.h"
#include "camera.h"
#include "object.h"
#include "mathutils.h"
#include "viszutils.h"
#include "frame.h"
#include "keyframe.h"
#include "mapdatabase.h"
#include "semanticmapper.h"
#include "trajectorysmoother.h"

#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>
#include <thread>


namespace eventobjectslam {

void LoadDetectionsWithFacet(
    const std::vector<std::string>& sDetections,
    std::vector<TwoDBoundingBox>& leftDetections,
    std::vector<TwoDBoundingBox>& rightDetections,
    std::shared_ptr<object::ObjectBase> pFacetObject    
);

void LoadDetections(
    const std::vector<std::string>& sDetections,
    std::vector<TwoDBoundingBox>& leftDetections,
    std::vector<TwoDBoundingBox>& rightDetections,
    const unsigned int imageWidth,
    const unsigned int imageHeight,
    std::shared_ptr<object::ObjectBase> pColorcone
);


void SaveOptimizedTraj(
    const std::string datasetRoot,
    std::vector<std::shared_ptr<Frame>> pFrameStack,
    const std::vector<Mat44Unaligned_t>& smoothedTrajectoryInWorld,
    const Mat44_t worldInReadworldTransform
);

void SaveLandmarks(
    const std::string datasetRoot,
    std::shared_ptr<MapDataBase> mapDb,
    const Mat44_t worldInReadworldTransform
);


class SystemConfig {
public:
    //! Constructor
    SystemConfig(const std::string& configFilePath);
    SystemConfig(const SystemConfig& config);

    //! Destructor
    ~SystemConfig(){};

    //! path to config file
    std::string _configFilePath;

    //! json node
    rapidjson::Document _jsonConfigNode;

    //! tracker parameters loaded from "tracker" section of the config
    TrackerParams _trackerParams;
};

class SLAMSystem {

public:
    SLAMSystem(const std::shared_ptr<SystemConfig>& cfg);

    ~SLAMSystem(){};

    void Startup();

    void InitializeCameraAndTracker(
        const unsigned int cameraID,
        const unsigned int cols,
        const unsigned int rows,
        const Eigen::Matrix3f kk,
        const float baseline,
        const std::string& debugPath
    );

    const Mat44_t UpdateOneFrame(
        const std::string& timestamp,
        const std::vector<std::string>& sDetections,
        std::shared_ptr<object::ObjectBase> pTheLandmarkObject,
        const bool isDebug
    );

    std::shared_ptr<SystemConfig> _cfg;
    std::shared_ptr<MapDataBase> _pMapDb;
    std::unique_ptr<SemanticMapper> _pMapper;
    // localBA and landmark pruning thread
    std::unique_ptr<std::thread> _pMapperThread = nullptr;

    std::vector<std::shared_ptr<Frame>> _allFramesStack;  // Including frames that did not contain target objects.
    std::vector<Mat44Unaligned_t> _smoothedTrajectoryInWorld;  // one smoothed world pose per frame in _allFramesStack.
    std::vector<std::shared_ptr<Frame>> _pFrameStack;
    std::vector<std::shared_ptr<KeyFrame>> _pKeyFrameStack;

    Mat44_t nextFrameInCameraTransform;  // Note: camera velocity

private:
    // compose the stereo debug view (input frames + detections + tracking status) and publish it to _pMapDb.
    // keyframeInsertReason: non-empty when a keyframe was inserted on this frame; shown on the image.
    // trackingMethod: name of the tracking method that succeeded on this frame; shown on the image.
    void UpdateDebugView(
        const std::shared_ptr<Frame>& pFrame,
        const bool isTrackingSuccess,
        const Mat44_t& framePoseInWorld,
        const std::string& keyframeInsertReason,
        const std::string& trackingMethod
    );

    std::shared_ptr<camera::CameraBase> _camera = nullptr;
    std::unique_ptr<FrameTracker> _frameTracker = nullptr;
    std::string _sStereoSequencePathForDebug;
    size_t _maxNumDetectionsInHistory = 0;  // Note: the most detections ever seen in one frame during the run.
    // frames whose tracking failed since the last success; their poses get linearly interpolated at the next success.
    std::vector<std::shared_ptr<Frame>> _framesPendingInterpolation;
    // Note: DontAlign avoids imposing Eigen over-alignment on the (stack-allocated) SLAMSystem object.
    Eigen::Matrix<float, 4, 4, Eigen::DontAlign> _lastSuccessfulFrameWorldPose = Eigen::Matrix4f::Identity();
    // online kalman smoothing of the returned per-frame pose (port of tool_evalAndViszData/smooth_traj.py).
    TrajectorySmoother _trajectorySmoother;

};

}


#endif  // EVENTOBJECTSLAM_SYSTEM_H