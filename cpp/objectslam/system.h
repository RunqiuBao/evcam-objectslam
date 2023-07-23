#ifndef EVENTOBJECTSLAM_SYSTEM_H
#define EVENTOBJECTSLAM_SYSTEM_H

#include "objectslam.h"
#include "frametracker.h"
#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>


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
    std::shared_ptr<SystemConfig> _cfg;

    SLAMSystem(const std::shared_ptr<SystemConfig>& cfg)
    :_cfg(cfg)
    {}

    ~SLAMSystem(){};

    void TestTrackStereoSequence(const std::string stereoSequencePath);

    Mat44_t AppendStereoFrame(const cv::Mat& leftImg, const cv::Mat& rightImg, const double timestamp, const cv::Mat& maskImg);


private:
    std::shared_ptr<camera::CameraBase> _camera = nullptr;
    std::shared_ptr<FrameTracker> _frameTracker = nullptr;


};

}


#endif  // EVENTOBJECTSLAM_SYSTEM_H