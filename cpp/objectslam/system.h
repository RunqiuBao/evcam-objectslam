#ifndef EVENTOBJECTSLAM_SYSTEM_H
#define EVENTOBJECTSLAM_SYSTEM_H

#include "objectslam.h"
#include "frametracker.h"
#include "semanticmapper.h"
#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>
#include <thread>


namespace eventobjectslam {

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

    void TestTrackStereoSequence(const std::string stereoSequencePath);

    Mat44_t AppendStereoFrame(const cv::Mat& leftImg, const cv::Mat& rightImg, const double timestamp, const cv::Mat& maskImg);

    std::shared_ptr<SystemConfig> _cfg;
    std::shared_ptr<MapDataBase> _pMapDb;
    std::unique_ptr<SemanticMapper> _pMapper;
    // localBA and landmark pruning thread
    std::unique_ptr<std::thread> _pMapperThread = nullptr;

private:
    std::shared_ptr<camera::CameraBase> _camera = nullptr;
    std::unique_ptr<FrameTracker> _frameTracker = nullptr;

};

}


#endif  // EVENTOBJECTSLAM_SYSTEM_H