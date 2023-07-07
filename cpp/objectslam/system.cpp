#include "system.h"
#include <filesystem>

#include <logging.h>
TDO_LOGGER("eventobjectslam.system")

using namespace std;

namespace eventobjectslam {

SystemConfig::SystemConfig(const std::string& configFilePath){
    _configFilePath = configFilePath;
    // Open the JSON file.
    FILE* fp = fopen(_configFilePath.c_str(), "rb");
    char readBuffer[65536];
    rapidjson::FileReadStream frs(fp, readBuffer,
                                    sizeof(readBuffer));
    _jsonConfigNode.ParseStream(frs);
    fclose(fp);
}

SystemConfig::SystemConfig(const SystemConfig& config){
    _configFilePath = config._configFilePath;
    _jsonConfigNode.CopyFrom(config._jsonConfigNode, _jsonConfigNode.GetAllocator());
}

void SLAMSystem::TestTrackStereoSequence(const std::string sStereoSequencePath){
    // the dataset dir includes `leftcam` and `rightcam` and `colorconeInfo.json`
    // in each cam folder, it includes `*.png`, `detectionId*/(yolos, including linemod based template selection)`
    
    // relocalization: when relocalization happens, assume it always see old objects.
    filesystem::path stereoSequencePath = sStereoSequencePath;
    for (const filesystem::directory_entry& oneFilePath : filesystem::directory_iterator(stereoSequencePath)) {
        if (filesystem::is_regular_file(oneFilePath) && oneFilePath.path().extension() == ".png") {
            TDO_LOG_VERBOSE(oneFilePath.path().string());
        }
    }
}

}