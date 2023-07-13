#include "system.h"
#include "camera.h"
#include <filesystem>

#include <logging.h>
TDO_LOGGER("eventobjectslam.system")


namespace eventobjectslam {

SystemConfig::SystemConfig(const std::string& configFilePath){
    _configFilePath = configFilePath;
    // Open the JSON file.
    FILE* fp = fopen(_configFilePath.c_str(), "rb");
    char readBuffer[65536];
    rapidjson::FileReadStream frs(
        fp,
        readBuffer,
        sizeof(readBuffer)
    );
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

    rapidjson::Document& sysConfigJson = _cfg->_jsonConfigNode;
    Eigen::Matrix3f kk;
    rapidjson::Value& kkJson = sysConfigJson["camera"]["kk"];
    for (rapidjson::SizeType i = 0; i < kkJson.Size(); i++){
        rapidjson::Value& kkJsonRow = kkJson[i];
        for (rapidjson::SizeType j = 0; j < kkJsonRow.Size(); j++){
            kk(static_cast<int>(i), static_cast<int>(j)) = kkJsonRow[j].GetFloat();
        }
    }

    camera::CameraBase myStereoCamera(
        sysConfigJson["camera"]["imageWidth"].GetInt(),
        sysConfigJson["camera"]["imageHeight"].GetInt(),
        kk,
        sysConfigJson["camera"]["baseline"].GetFloat()
    );

    std::filesystem::path leftCamPath = sStereoSequencePath;
    leftCamPath.append("leftcam/");
    TDO_LOG_DEBUG("entered test track");
    std::vector<std::string> filenames;
    for (const std::filesystem::directory_entry& oneFilePath : std::filesystem::directory_iterator(leftCamPath)) {
        if (std::filesystem::is_regular_file(oneFilePath) && oneFilePath.path().extension() == ".png") {
            filenames.push_back(oneFilePath.path().filename().stem());
        }
    }
    std::sort(filenames.begin(), filenames.end());

    for(const std::string& filename : filenames){
        TDO_LOG_DEBUG(filename);

        // stereo triangulation and template scale based filtering
        // ransac based plane estimation
        // at the same time, ground plane based detection filtering
        // object 3d bounding box detection and ground plane based pose correction

        // input the correct detections to frameTracker, get pose return from frameTracker and print.
    }
}

}
