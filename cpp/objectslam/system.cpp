#include "system.h"
#include "camera.h"
#include "object.h"
#include "mathutils.h"
#include "viszutils.h"
#include "frame.h"
#include "keyframe.h"

#include <filesystem>
#include <fstream>
#include <vector>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cassert>

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

void LoadDetections(const std::vector<std::string>& sDetections, std::vector<TwoDBoundingBox>& detections, const unsigned int imageWidth, const unsigned int imageHeight, std::shared_ptr<object::ObjectBase> pColorcone){
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
        detections.push_back(TwoDBoundingBox(x, y, bWidth, bHeight, templateID, templateScale, pColorcone));
        TDO_LOG_DEBUG_FORMAT("one detection: %f, %f, template No. %d, at scale %f", x % y % templateID % templateScale);
    }
}

// Mat44_t SLAMSystem::AppendStereoFrame(const cv::Mat& leftImg, const cv::Mat& rightImg, const double timestamp, const cv::Mat& maskImg) {
//     // TODO:  need to form data::frame before give to tracker
//     const Mat44_t cam_pose_cw = _frameTracker->TrackStereoImage(leftImg, rightImg, timestamp, maskImg);
//     return cam_pose_cw;
// }

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
    _camera = std::make_shared<camera::CameraBase>(myStereoCamera);
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
    std::shared_ptr<object::ObjectBase> pColorcone = std::make_shared<object::ObjectBase>(colorcone);

    Mat44_t cameraInWorldTransform = Eigen::Matrix4f::Identity();
    Mat44_t nextFrameInCameraTransform = Eigen::Matrix4f::Identity();
    Eigen::Quaternionf q;
    q.x() = -0.49999999999999956;
    q.y() = 0.5000000218556936;
    q.z() = 0.49999999999999956;
    q.w() = -0.49999997814430636; 
    Eigen::Matrix3f rWorldToRealWorld = q.toRotationMatrix();
    Mat44_t worldInRealWorld = Eigen::Matrix4f::Identity();
    worldInRealWorld.block(0, 0, 3, 3) = rWorldToRealWorld;
    worldInRealWorld.block(0, 2, 3, 1) *= -1;
    TDO_LOG_DEBUG("rWorldToRealWorld: " << rWorldToRealWorld);
    Eigen::Vector3f tWorldToRealWorld(-9.255331993103027, 7.211221218109131, 2.187476634979248);
    worldInRealWorld.block(0, 3, 3, 1) = tWorldToRealWorld;

    std::filesystem::path trackResultPath = sStereoSequencePath;
    trackResultPath.append("cameraTrack.txt");
    std::ofstream outputFile(trackResultPath.string());
    int frameCount = 0;

    FrameTracker _tracker(_camera);
    _tracker._sStereoSequencePathForDebug = sStereoSequencePath;
    std::vector<Frame> _frameStack;
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
        LoadDetections(sDetections, leftCamDetections, myStereoCamera._cols, myStereoCamera._rows, pColorcone);

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
        LoadDetections(sDetections, rightCamDetections, myStereoCamera._cols, myStereoCamera._rows, pColorcone);

        Frame oneFrame(FrameType::Stereo, static_cast<double>(frameCount), _camera);
        oneFrame.SetDetectionsFromExternalSrc(std::move(leftCamDetections), std::move(rightCamDetections));

        std::vector<std::shared_ptr<TwoDBoundingBox>> matchedLeftCamDetections, matchedRightCamDetections;
        auto matchedDetections = oneFrame.GetMatchedDetections();  // Note: return a tuple.
        matchedLeftCamDetections = std::get<0>(matchedDetections);
        matchedRightCamDetections = std::get<1>(matchedDetections);

        std::filesystem::path leftCamImagePath = sStereoSequencePath;
        leftCamImagePath.append("leftcam/").append(filename + ".png");
        cv::Mat display3DDetections = cv::imread(leftCamImagePath.string(), cv::IMREAD_GRAYSCALE);
        std::vector<ThreeDDetection> threeDDetections = oneFrame._threeDDetections;
        int countThreeDDetection = 0;
        cv::Mat testBinaryTracking(myStereoCamera._rows, myStereoCamera._cols, CV_8UC1, cv::Scalar(0));
        for (ThreeDDetection oneDetection : threeDDetections){
            Eigen::Matrix<float, 2, Eigen::Dynamic> dstPoints = Eigen::Matrix<float, 2, Eigen::Dynamic>::Zero(2, oneDetection._vertices3DInCamera.cols());
            myStereoCamera.ProjectPoints(oneDetection._vertices3DInCamera, dstPoints);
            viszutils::Draw3DBoundingBox(dstPoints, display3DDetections);
            // for debug binary tracking
            Eigen::MatrixXf detectionPoseCenter = Eigen::MatrixXf::Zero(3, 1);
            detectionPoseCenter.block(0, 0, 3, 1) = oneDetection._objectInCameraTransform.block(0, 3, 3, 1);
            std::vector<cv::Point> point2DPoseCenter = mathutils::ProjectPoints3DToPoints2D(detectionPoseCenter, myStereoCamera);
            TDO_LOG_DEBUG("detection " << std::to_string(countThreeDDetection) << ", pose center 2d: " << point2DPoseCenter[0]);
            cv::circle(testBinaryTracking, point2DPoseCenter[0], 5, cv::Scalar(255), -1, cv::LINE_AA);

            countThreeDDetection++;
        }
        std::filesystem::path testBinaryTrackingPath = sStereoSequencePath;
        testBinaryTrackingPath.append("testBinaryTracking/").append(filename + ".png");
        cv::imwrite(testBinaryTrackingPath.string() , testBinaryTracking * 255);
        oneFrame._leftTrackImage = testBinaryTracking * 255;
        std::filesystem::path debug3DDetectionPath = sStereoSequencePath;
        debug3DDetectionPath.append("debug3DDetection/").append(filename + ".png");
        cv::imwrite(debug3DDetectionPath.string() , display3DDetections);

        // // ransac based plane estimation
           // at the same time, ground plane based detection filtering

        // // input the correct detections to frameTracker, get pose return from frameTracker and print.

        if (filename == "000000"){
            oneFrame.SetPose(Eigen::Matrix4f::Identity());
            std::shared_ptr<KeyFrame> pOneKeyframe = std::make_shared<KeyFrame>(oneFrame);  // Note: allocated on heap. will not disappear due to out of scope.
            _tracker._pRefKeyframe = pOneKeyframe;
            oneFrame._pRefKeyframe = pOneKeyframe;
            oneFrame.SetDetectionsAsLandmarks();
        }
        else{
            bool isSuccess = _tracker.DoMotionBasedTrack(oneFrame, _frameStack.back(), nextFrameInCameraTransform);
            if (!isSuccess){
                bool isSuccess = _tracker.Do2DTrackingBasedTrack(oneFrame, _frameStack.back(), nextFrameInCameraTransform);
            }

        }
        TDO_LOG_DEBUG("------------- End of one frame ------------");
        Mat44_t cameraInRealWorld = worldInRealWorld * oneFrame.GetPose();
        Eigen::Quaternionf myQuaternion(cameraInRealWorld.block<3, 3>(0, 0));
        outputFile << std::to_string(frameCount) << " " << cameraInRealWorld(0, 3) << " " << cameraInRealWorld(1, 3) << " " << cameraInRealWorld(2, 3) << " " << myQuaternion.x() << " " << myQuaternion.y() << " " << myQuaternion.z() << " " << myQuaternion.w() << std::endl;
        TDO_LOG_DEBUG("cameraInRealWorld: \n" << cameraInRealWorld);
        frameCount++;
        _frameStack.push_back(oneFrame);

        if (filename == "000036"){
            break;
        }

    }
    outputFile.close();
}

}
