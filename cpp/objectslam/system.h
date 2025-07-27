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
    std::vector<std::shared_ptr<Frame>> _pFrameStack;
    std::vector<std::shared_ptr<KeyFrame>> _pKeyFrameStack;

    Mat44_t nextFrameInCameraTransform;  // Note: camera velocity

private:
    std::shared_ptr<camera::CameraBase> _camera = nullptr;
    std::unique_ptr<FrameTracker> _frameTracker = nullptr;

};

}


#endif  // EVENTOBJECTSLAM_SYSTEM_H