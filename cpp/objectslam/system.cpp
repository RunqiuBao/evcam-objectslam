#include "system.h"
#include "camera.h"
#include "object.h"

#include <filesystem>
#include <fstream>
#include <vector>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <opencv2/opencv.hpp>

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

void LoadDetections(const std::vector<std::string>& sDetections, std::vector<TwoDBoundingBox>& detections, const unsigned int imageWidth, const unsigned int imageHeight){
    detections.reserve(sDetections.size());
    for (std::string sDetection : sDetections){
        std::vector<std::string> splitSDetection;
        boost::split(splitSDetection, sDetection, boost::is_any_of(" "));
        float x, y, bWidth, bHeight, templateScale;
        int templateID;
        x = boost::lexical_cast<float>(splitSDetection[1].c_str()) * imageWidth;
        y = boost::lexical_cast<float>(splitSDetection[2].c_str()) * imageHeight;
        bWidth = boost::lexical_cast<float>(splitSDetection[3].c_str()) * imageWidth;
        bHeight = boost::lexical_cast<float>(splitSDetection[4].c_str()) * imageHeight;
        templateID = boost::lexical_cast<size_t>(splitSDetection[5].c_str());
        splitSDetection[6].erase(
            std::remove_if(splitSDetection[6].begin(), 
            splitSDetection[6].end(),
            [](unsigned char x) { return x == '\n'; }),
            splitSDetection[6].end()
        );
        templateScale = boost::lexical_cast<float>(splitSDetection[6].c_str());
        detections.push_back(TwoDBoundingBox(x, y, bWidth, bHeight, templateID, templateScale));
        TDO_LOG_DEBUG_FORMAT("one detection: %f, %f, template No. %d, at scale %f", x % y % templateID % templateScale);
    }
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
        0,
        sysConfigJson["camera"]["imageWidth"].GetInt(),
        sysConfigJson["camera"]["imageHeight"].GetInt(),
        kk,
        sysConfigJson["camera"]["baseline"].GetFloat()
    );
    TDO_LOG_DEBUG_FORMAT("myStereoCamera imageWidth: %d", myStereoCamera._cols);

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

    std::filesystem::path templatesPath = sStereoSequencePath;
    templatesPath.append("templates/");
    object::ObjectBase colorcone(templatesPath.string());

    for(const std::string& filename : filenames){
        TDO_LOG_DEBUG(filename);

        std::filesystem::path leftCamPath = sStereoSequencePath;
        leftCamPath.append("leftcam/");
        std::ifstream detectionResult(leftCamPath.append("detectionID0").append(filename + ".txt"));
        if (!detectionResult.is_open()) {
            TDO_LOG_DEBUG("Failed to open the left detectionResult (" << filename << ").");
            continue;
        }
        std::vector<std::string> sDetections;
        std::string sDetection;
        while (std::getline(detectionResult, sDetection)) {
            sDetections.push_back(sDetection); // Store each line in the vector
        }
        detectionResult.close();
        std::vector<TwoDBoundingBox> leftCamDetections;
        LoadDetections(sDetections, leftCamDetections, myStereoCamera._cols, myStereoCamera._rows);

        std::filesystem::path rightCamPath = sStereoSequencePath;
        rightCamPath.append("rightcam/");
        std::ifstream detectionResultRightCam(rightCamPath.append("detectionID0").append(filename + ".txt"));
        if (!detectionResultRightCam.is_open()) {
            TDO_LOG_DEBUG("Failed to open the right detectionResult (" << filename << ").");
            continue;
        }
        sDetections.clear();
        while (std::getline(detectionResultRightCam, sDetection)) {
            sDetections.push_back(sDetection); // Store each line in the vector
        }
        detectionResultRightCam.close();
        std::vector<TwoDBoundingBox> rightCamDetections;
        LoadDetections(sDetections, rightCamDetections, myStereoCamera._cols, myStereoCamera._rows);

        // // stereo triangulation and template scale based filtering
        std::vector<std::shared_ptr<TwoDBoundingBox>> matchedLeftCamDetections, matchedRightCamDetections;
        myStereoCamera.MatchStereoBBoxes(leftCamDetections, rightCamDetections, colorcone, matchedLeftCamDetections, matchedRightCamDetections);
        std::vector<ThreeDDetection> threeDDetections;
        myStereoCamera.CreateThreeDDetections(matchedLeftCamDetections, matchedRightCamDetections, colorcone, threeDDetections);

        std::filesystem::path leftCamImagePath = sStereoSequencePath;
        leftCamImagePath.append("leftcam/").append(filename + ".png");
        cv::Mat display3DDetections = cv::imread(leftCamImagePath.string(), cv::IMREAD_GRAYSCALE);
        for (ThreeDDetection oneDetection : threeDDetections){
            
        }

        // // ransac base plane estimation
           // at the same time, ground plane based detection filtering
        // // object 3d bounding box detection and ground plane based pose correction
        // // input the correct detections to frameTracker, get pose return from frameTracker and print.

        if (filename == "000006"){
            break;
        }
    }
}

}
