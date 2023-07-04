#ifndef EVENTOBJECTSLAM_SYSTEM_H
#define EVENTOBJECTSLAM_SYSTEM_H

#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>

namespace eventobjectslam {

class SystemConfig {
public:
    //! Constructor
    SystemConfig(const std::string& configFilePath){
        _configFilePath = configFilePath;
        // Open the JSON file.
        rapidjson::FileReadStream fstream(_configFilePath);
        _jsonConfigNode.ParseStream(fstream);
    }

    //! Destructor
    ~SystemConfig();

    //! path to config file
    const std::string _configFilePath;

    //! json node
    const rapidjson::Document _jsonConfigNode;
};

class SLAMSystem {

public:
    const std::shared_ptr<SystemConfig> _cfg;

    SLAMSystem(const std::shared_ptr<SystemConfig>& cfg);

    ~SLAMSystem();

    void TestTrackStereoSequence(
        const std::string stereoSequencePath
    );
};

}


#endif  // EVENTOBJECTSLAM_SYSTEM_H