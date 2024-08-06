#include "system.h"
#include "camera.h"
#include "object.h"
#include "mathutils.h"
#include "viszutils.h"
#include "frame.h"
#include "keyframe.h"
#include "mapdatabase.h"
#include "semanticmapper.h"

#include <filesystem>
#include <functional>
#include <fstream>
#include <vector>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cassert>
#include <chrono>

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

SLAMSystem::SLAMSystem(const std::shared_ptr<SystemConfig>& cfg)
    :_cfg(cfg)
{
    _pMapDb = std::make_shared<MapDataBase>();
    _pMapper = std::make_unique<SemanticMapper>(_pMapDb);
}

void SLAMSystem::Startup() {
    TDO_LOG_DEBUG("Startup SLAM system.");
    _pMapperThread  = std::unique_ptr<std::thread>(new std::thread(&SemanticMapper::Run, _pMapper.get()));
}

void LoadDetections(
    const std::vector<std::string>& sDetections,
    std::vector<TwoDBoundingBox>& leftDetections,
    std::vector<TwoDBoundingBox>& rightDetections,
    const unsigned int imageWidth,
    const unsigned int imageHeight,
    std::shared_ptr<object::ObjectBase> pColorcone
){
    leftDetections.reserve(sDetections.size());
    rightDetections.reserve(sDetections.size());
    for (std::string sDetection : sDetections){
        std::vector<std::string> splitSDetection;
        boost::split(splitSDetection, sDetection, boost::is_any_of(" "));
        float x, y, bWidth, bHeight, x_r, y_r, bWidth_r, bHeight_r, detectionScore;
        x = boost::lexical_cast<float>(splitSDetection[1].c_str()) * imageWidth;
        y = boost::lexical_cast<float>(splitSDetection[2].c_str()) * imageHeight;
        bWidth = boost::lexical_cast<float>(splitSDetection[3].c_str()) * imageWidth;
        bHeight = boost::lexical_cast<float>(splitSDetection[4].c_str()) * imageHeight;
        x_r = boost::lexical_cast<float>(splitSDetection[5].c_str()) * imageWidth;
        y_r = boost::lexical_cast<float>(splitSDetection[6].c_str()) * imageHeight;
        bWidth_r = boost::lexical_cast<float>(splitSDetection[7].c_str()) * imageWidth;
        bHeight_r = boost::lexical_cast<float>(splitSDetection[8].c_str()) * imageHeight;
        Vec2_t keypt1(
            x - bWidth / 2 + boost::lexical_cast<float>(splitSDetection[9].c_str()) * bWidth,
            y - bHeight / 2 + boost::lexical_cast<float>(splitSDetection[10].c_str()) * bHeight
        );
        Vec2_t keypt2(
            x - bWidth / 2 + boost::lexical_cast<float>(splitSDetection[11].c_str()) * bWidth,
            y - bHeight / 2 + boost::lexical_cast<float>(splitSDetection[12].c_str()) * bHeight
        );
        std::vector<Vec2_t> keypts = {keypt1, keypt2};
        detectionScore = boost::lexical_cast<float>(splitSDetection[13].c_str());

        leftDetections.push_back(TwoDBoundingBox(x, y, bWidth, bHeight, pColorcone, detectionScore, keypts));
        Vec2_t keypt1_r(
            x_r - bWidth_r / 2 + boost::lexical_cast<float>(splitSDetection[9].c_str()) * bWidth_r,
            y_r - bHeight / 2 + boost::lexical_cast<float>(splitSDetection[10].c_str()) * bHeight
        );
        Vec2_t keypt2_r(
            x - bWidth / 2 + boost::lexical_cast<float>(splitSDetection[11].c_str()) * bWidth,
            y - bHeight / 2 + boost::lexical_cast<float>(splitSDetection[12].c_str()) * bHeight
        );
        std::vector<Vec2_t> keypts_r = {keypt1_r, keypt2_r};
        rightDetections.push_back(TwoDBoundingBox(x_r, y, bWidth_r, bHeight, pColorcone, detectionScore, keypts_r));
        TDO_LOG_DEBUG_FORMAT("one 2d detection (confi %f): %f, %f", detectionScore % x % y);
    }
}

// Mat44_t SLAMSystem::AppendStereoFrame(const cv::Mat& leftImg, const cv::Mat& rightImg, const double timestamp, const cv::Mat& maskImg) {
//     // TODO:  need to form data::frame before give to tracker
//     const Mat44_t cam_pose_cw = _frameTracker->TrackStereoImage(leftImg, rightImg, timestamp, maskImg);
//     return cam_pose_cw;
// }

void SaveOptimizedTraj(const std::string datasetRoot, std::vector<std::shared_ptr<Frame>> pFrameStack, const Mat44_t worldInReadworldTransform){
    std::filesystem::path trajFilePath = datasetRoot;
    trajFilePath.append("cameraTrackOptimized.txt");
    std::filesystem::path trajMaskFilePath = datasetRoot;
    trajMaskFilePath.append("cameraTrackOptimized_mask.txt");
    std::ofstream trajFile(trajFilePath.string());
    std::ofstream trajMaskFile(trajMaskFilePath.string());

    size_t frameCount = 0;
    for (auto pFrame : pFrameStack) {
        if (pFrame->_isTracked) {
            Mat44_t frameInRealWorld = worldInReadworldTransform * pFrame->_pRefKeyframe->GetKeyframePoseInWorld() * pFrame->GetPose();
            Eigen::Quaternionf myQuaternion(frameInRealWorld.block<3, 3>(0, 0));
            trajFile << std::to_string(frameCount) << " " << frameInRealWorld(0, 3) << " " << frameInRealWorld(1, 3) << " " << frameInRealWorld(2, 3) << " " << myQuaternion.x() << " " << myQuaternion.y() << " " << myQuaternion.z() << " " << myQuaternion.w() << std::endl;
            trajMaskFile << "1" << std::endl;
        }
        else {
            trajFile << std::to_string(frameCount) << " 0 0 0 0 0 0 1" << std::endl;
            trajMaskFile << "0" << std::endl;
        }
        frameCount++;
    }
    
    TDO_LOG_DEBUG("saved " << std::to_string(frameCount) << " frames pose. (optimized)");

    trajFile.close();
    trajMaskFile.close();
}

void SaveLandmarks(const std::string datasetRoot, std::shared_ptr<MapDataBase> mapDb, const Mat44_t worldInReadworldTransform) {
    std::filesystem::path landmarksFilePath = datasetRoot;
    landmarksFilePath.append("landmarks.txt");
    std::ofstream landmarksFile(landmarksFilePath.string());

    size_t landmarkCount = 0;
    for (auto id_pLandmark : mapDb->_landmarks) {
        Mat44_t landmarkPoseInWorld = id_pLandmark.second->GetLandmarkPoseInWorld();
        Mat44_t landmarkPoseInRealWorld = worldInReadworldTransform * landmarkPoseInWorld;
        landmarksFile << std::to_string(id_pLandmark.first) << " " << landmarkPoseInRealWorld(0, 3) << " " << landmarkPoseInRealWorld(1, 3) << " " << landmarkPoseInRealWorld(2, 3) << " " << landmarkPoseInRealWorld(0, 2) << " " << landmarkPoseInRealWorld(1, 2) << " " << landmarkPoseInRealWorld(2, 2) << std::endl;
        landmarkCount++;
    }
    TDO_LOG_DEBUG("saved " << std::to_string(landmarkCount) << " landmarks. (optimized)");
    landmarksFile.close();
}

void SLAMSystem::TestTrackStereoSequence(const std::string sStereoSequencePath){
    // the dataset dir includes `detections` and `sysconfig.json`

    rapidjson::Document& sysConfigJson = _cfg->_jsonConfigNode;

    Vec3_t objectSize;
    objectSize << sysConfigJson["objects"]["0"]["objectSize"][0].GetFloat(),
                  sysConfigJson["objects"]["0"]["objectSize"][1].GetFloat(),
                  sysConfigJson["objects"]["0"]["objectSize"][2].GetFloat();
    _pMapDb->SetObjectSize(objectSize);

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

    std::filesystem::path detectionsPath = sStereoSequencePath;
    detectionsPath.append("detections");
    TDO_LOG_DEBUG("entered test track");

    std::vector<std::string> filenames;
    for (const std::filesystem::directory_entry& oneFilePath : std::filesystem::directory_iterator(detectionsPath)) {
        if (std::filesystem::is_regular_file(oneFilePath) && oneFilePath.path().extension() == ".txt") {
            filenames.push_back(oneFilePath.path().filename().stem());
        }
    }
    std::sort(filenames.begin(), filenames.end());

    std::string objectName = sysConfigJson["objects"]["0"]["objectName"].GetString();
    object::ObjectBase colorcone(objectName);
    std::shared_ptr<object::ObjectBase> pColorcone = std::make_shared<object::ObjectBase>(colorcone);

    Mat44_t cameraInWorldTransform = Eigen::Matrix4f::Identity();
    Mat44_t nextFrameInCameraTransform = Eigen::Matrix4f::Identity();

    // // seq0, seq2
    // Eigen::Quaternionf q;
    // q.x() = -0.49999999999999956;
    // q.y() = 0.5000000218556936;
    // q.z() = 0.49999999999999956;
    // q.w() = -0.49999997814430636; 
    // Eigen::Matrix3f rWorldToRealWorld = q.toRotationMatrix();
    // Mat44_t worldInRealWorld = Eigen::Matrix4f::Identity();
    // worldInRealWorld.block(0, 0, 3, 3) = rWorldToRealWorld;
    // worldInRealWorld.block(0, 2, 3, 1) *= -1;
    // TDO_LOG_DEBUG("rWorldToRealWorld: " << rWorldToRealWorld);
    // Eigen::Vector3f tWorldToRealWorld(-9.255331993103027, 7.211221218109131, 2.187476634979248);
    // worldInRealWorld.block(0, 3, 3, 1) = tWorldToRealWorld;
    
    // //seq1
    // Mat44_t worldInRealWorld;
    // worldInRealWorld << -4.371139e-8, 1.736483e-1, -9.848077e-1, 4.,
    //                     1., 7.590408e-9,-4.304732e-8, -2.2000,
    //                     -0., -9.848077e-1, -1.736483e-1, 2.3650,
    //                     0., 0., 0., 1.;

    Mat44_t worldInRealWorld = Eigen::Matrix4f::Identity();

    std::filesystem::path trackResultPath = sStereoSequencePath;
    trackResultPath.append("cameraTrackRealTime.txt");
    std::ofstream outputFile(trackResultPath.string());
    
    int frameCount = 0;
    size_t keyframeCount = 0;  // Note: for pruning keyframe count.
    float minDistanceToCreateKeyframe = 0.2;  // inline parameter

    FrameTracker _tracker(_camera);
    _tracker._sStereoSequencePathForDebug = sStereoSequencePath;
    std::vector<std::shared_ptr<Frame>> _pFrameStack;
    std::vector<std::shared_ptr<Frame>> _allFramesStack;
    std::vector<std::shared_ptr<KeyFrame>> _pKeyFrameStack;
    Mat44_t cameraInRealWorld = Eigen::Matrix4f::Identity();  // Note: Pose for comparing to ground truth

    // for debug purpose
    bool isDebug = true;
    std::vector<size_t> numFramesEachKeyframe;
    for(const std::string& filename : filenames){
        TDO_LOG_DEBUG(filename);

        auto starttime = std::chrono::steady_clock::now();
        std::shared_ptr<Frame> pOneFrame = std::make_shared<Frame>(FrameType::Stereo, static_cast<double>(frameCount), _camera);
        _allFramesStack.push_back(pOneFrame);

        // load detection results from txt file.
        std::filesystem::path detectionFilePath = detectionsPath;
        std::ifstream detectionResult(detectionFilePath.append(filename + ".txt"));
        TDO_LOG_DEBUG_FORMAT("opening detection: %s", detectionFilePath.string());
        if (!detectionResult.is_open()) {
            TDO_LOG_DEBUG("Failed to open the detectionResult (" << filename << ").");
            Eigen::Quaternionf myQuaternion(cameraInRealWorld.block<3, 3>(0, 0));
            outputFile << std::to_string(frameCount) << " " << cameraInRealWorld(0, 3) << " " << cameraInRealWorld(1, 3) << " " << cameraInRealWorld(2, 3) << " " << myQuaternion.x() << " " << myQuaternion.y() << " " << myQuaternion.z() << " " << myQuaternion.w() << std::endl;
            frameCount++;
            continue;
        }
        std::vector<std::string> sDetections;
        std::string sDetection;
        while (std::getline(detectionResult, sDetection)) {
            sDetections.push_back(sDetection); // Store each line in the vector
        }
        detectionResult.close();
        TDO_LOG_DEBUG_FORMAT("length of sDetection: %d", sDetections.size());
        if (sDetections.size() == 0){
            continue;
        }

        std::vector<TwoDBoundingBox> leftCamDetections, rightCamDetections;  // Note: load left and right detections and list correspondingly.
        LoadDetections(sDetections, leftCamDetections, rightCamDetections, myStereoCamera._cols, myStereoCamera._rows, pColorcone);

        pOneFrame->SetDetectionsFromExternalSrc(std::move(leftCamDetections), std::move(rightCamDetections));

        std::vector<std::shared_ptr<TwoDBoundingBox>> matchedLeftCamDetections, matchedRightCamDetections;
        auto matchedDetections = pOneFrame->GetMatchedDetections();  // Note: return a tuple.
        matchedLeftCamDetections = std::get<0>(matchedDetections);
        matchedRightCamDetections = std::get<1>(matchedDetections);

        if (isDebug){
            std::filesystem::path leftCamImagePath = sStereoSequencePath;
            leftCamImagePath.append("concentrated/left/").append(filename + ".png");
            cv::Mat display3DDetections = cv::imread(leftCamImagePath.string(), cv::IMREAD_COLOR);
            std::vector<ThreeDDetection> threeDDetections = pOneFrame->_threeDDetections;
            int countThreeDDetection = 0;
            for (ThreeDDetection oneDetection : threeDDetections){
                Eigen::Matrix<float, 2, Eigen::Dynamic> dstPoints(2, 2);
                Eigen::Matrix<float, 3, Eigen::Dynamic> threeDKeypoints(3, 2);
                threeDKeypoints << oneDetection._keypt1InRefFrame(0), oneDetection._objectCenterInRefFrame(0),
                                   oneDetection._keypt1InRefFrame(1), oneDetection._objectCenterInRefFrame(1),
                                   oneDetection._keypt1InRefFrame(2), oneDetection._objectCenterInRefFrame(2);
                myStereoCamera.ProjectPoints(threeDKeypoints, dstPoints);
                viszutils::Draw5DDetections(dstPoints, display3DDetections);

                countThreeDDetection++;
            }

            std::filesystem::path debug3DDetectionPath = sStereoSequencePath;
            debug3DDetectionPath.append("debug3DDetections/");
            if (!std::filesystem::exists(debug3DDetectionPath) && !std::filesystem::create_directory(debug3DDetectionPath)){
                TDO_LOG_ERROR_FORMAT("Failed to create the folder: %s", debug3DDetectionPath.string());
                throw std::runtime_error("OS Error");
            }
            debug3DDetectionPath.append(filename + ".png");
            cv::imwrite(debug3DDetectionPath.string() , display3DDetections);
        }

        // // input the correct detections to frameTracker, get pose return from frameTracker and print.

        if (!_tracker.GetTrackerStatus()){
            pOneFrame->SetPose(Eigen::Matrix4f::Identity());
            pOneFrame->_isTracked = true;
            TDO_LOG_INFO("keyframe insert! frame number: " << filename);
            std::shared_ptr<KeyFrame> pOneKeyframe = std::make_shared<KeyFrame>(pOneFrame, Eigen::Matrix4f::Identity(), _camera);  // Note: allocated on heap. will not disappear due to out of scope.
            _tracker._pRefKeyframe = pOneKeyframe;
            _tracker.SetTrackerStatus(true);
            _pKeyFrameStack.push_back(pOneKeyframe);
            _pMapDb->AddKeyFrame(pOneKeyframe);
            pOneFrame->SetDetectionsAsRefObjects();
            _tracker.CreateNewLandmarks(pOneKeyframe, _pMapDb, isDebug);
        }
        else{
            Mat44_t nextFrameInCameraTransformBackup = nextFrameInCameraTransform;  // Note: backup in case first track fails and nextFrameInCameraTransform will be set to identity,
            bool isSuccess = _tracker.DoMotionBasedTrack(*pOneFrame, (*_pFrameStack.back()), nextFrameInCameraTransform, isDebug);

            // if (!isSuccess){
            //     // TODO: project all ref objects to previous frame then track.
            //     bool isSuccess = _tracker.Do2DTrackingBasedTrack(*pOneFrame, (*_pFrameStack.back()), nextFrameInCameraTransform, isDebug);
            // }

            if (!isSuccess){
                bool isSuccess = _tracker.DoDenseAlignmentBasedTrack(*pOneFrame, (*_pFrameStack.back()), isDebug);
                if (isSuccess){
                    nextFrameInCameraTransform = (*_pFrameStack.back()).GetPose().inverse() * pOneFrame->GetPose();
                }
            }

            if (!isSuccess){
                // try relocalize from map
                bool isSuccess = _tracker.DoRelocalizeFromMap(*pOneFrame, (*_pFrameStack.back()), _pMapDb, nextFrameInCameraTransform, isDebug);
                if (!isSuccess){
                    TDO_LOG_DEBUG("relocalization also failed.");
                }
            }

            if (pOneFrame->_frameID == 280) {
                TDO_LOG_DEBUG("baodebug");
            }
            
            // insert new key frame if detection increased than previous frame
            if (isSuccess && (pOneFrame->_threeDDetections.size() > (*_pFrameStack.back())._threeDDetections.size())){
            // bool bAddKeyframe = false;
            // if (_pFrameStack.size() < 2) {
            //     if (isSuccess && (pOneFrame->_threeDDetections.size() > (*_pFrameStack.back())._threeDDetections.size())) {
            //         bAddKeyframe = true;
            //     }
            // }
            // else if (_pFrameStack.size() >= 2){
            //     if (isSuccess
            //         && (pOneFrame->_threeDDetections.size() > (*_pFrameStack.back())._threeDDetections.size())
            //         && (pOneFrame->_threeDDetections.size() > (*_pFrameStack[_pFrameStack.size() - 2])._threeDDetections.size())
            //     ) {
            //         bAddKeyframe = true;
            //     }
            // }
            // if (bAddKeyframe){  // for vibration
                TDO_LOG_DEBUG_FORMAT("Last Keyframe(%d) contains %d frames.", _pKeyFrameStack.back()->_keyFrameID % _pKeyFrameStack.back()->_vFrames_ids.size());
                numFramesEachKeyframe.push_back(_pKeyFrameStack.back()->_vFrames_ids.size());  // debug code
                std::shared_ptr<KeyFrame> pOneKeyframe = std::make_shared<KeyFrame>(pOneFrame, _tracker._pRefKeyframe->GetKeyframePoseInWorld(), _camera);
                _tracker._pRefKeyframe = pOneKeyframe;
                _pMapDb->AddKeyFrame(pOneKeyframe);
                _pKeyFrameStack.push_back(pOneKeyframe);
                pOneFrame->SetDetectionsAsRefObjects();
                pOneFrame->SetPose(Eigen::Matrix4f::Identity());
                TDO_LOG_INFO("keyframe insert! frame number: " << filename);
                TDO_LOG_INFO("keyframe in world pose: " << pOneKeyframe->GetKeyframePoseInWorld());
                _tracker.CreateNewLandmarks(pOneKeyframe, _pMapDb, isDebug);
                // if (_pKeyFrameStack.size() > 5 && _pKeyFrameStack[_pKeyFrameStack.size() - 5]->_bContainNewLandmarks) {
                //     TDO_LOG_DEBUG_FORMAT("Entered landmark pruning at keyframe(%d).", _pKeyFrameStack[_pKeyFrameStack.size() - 5]->_keyFrameID);
                //     _pMapper->SchedulePruneLandmarksTask(_pKeyFrameStack[_pKeyFrameStack.size() - 5]);
                // }
                if (_pMapDb->_keyframes.size() - _pMapper->_numMinCovisibilityToPruneLandmark > keyframeCount) {
                    _pMapper->SchedulePruneLandmarksTask();
                    keyframeCount = _pMapDb->_keyframes.size();
                }
                // start BA
                while (!_pMapper->PushKeyframeForBA(pOneKeyframe)){
                    // Note: Force optimize every keyframe.
                    std::this_thread::sleep_for(std::chrono::microseconds(static_cast<unsigned int>(0.05 * 1e6)));
                }
            }

        }
        TDO_LOG_INFO("------------- End of one frame ------------");
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - starttime);
        TDO_LOG_INFO_FORMAT("frame(%s) tracking finished in %d milisec.", filename % duration.count());
        cameraInRealWorld = worldInRealWorld * _tracker._pRefKeyframe->GetKeyframePoseInWorld() * pOneFrame->GetPose();  //Note: think like there is a point in current frame, first it will be transformed into keyframe, then to world, then to realWorld.
        Eigen::Quaternionf myQuaternion(cameraInRealWorld.block<3, 3>(0, 0));
        outputFile << std::to_string(frameCount) << " " << cameraInRealWorld(0, 3) << " " << cameraInRealWorld(1, 3) << " " << cameraInRealWorld(2, 3) << " " << myQuaternion.x() << " " << myQuaternion.y() << " " << myQuaternion.z() << " " << myQuaternion.w() << std::endl;
        TDO_LOG_DEBUG("cameraInRealWorld: \n" << cameraInRealWorld);
        frameCount++;
        pOneFrame->_pRefKeyframe = _tracker._pRefKeyframe;
        _tracker._pRefKeyframe->_vFrames_ids[pOneFrame] = pOneFrame->_frameID;
        _pFrameStack.push_back(pOneFrame);
    }
    outputFile.close();
    // FIXME: the first keyframe will be optimized and bias from (0, 0, 0). Compensate it before saving.
    SaveOptimizedTraj(sStereoSequencePath, _allFramesStack, worldInRealWorld);
    SaveLandmarks(sStereoSequencePath, _pMapDb, worldInRealWorld);

    // print debug infos
    size_t minNumFrames = *std::min_element(numFramesEachKeyframe.begin(), numFramesEachKeyframe.end());
    size_t maxNumFrames = *std::max_element(numFramesEachKeyframe.begin(), numFramesEachKeyframe.end());
    TDO_LOG_INFO_FORMAT("keyframes contain maximum %d frames and minimum %d frames.", maxNumFrames % minNumFrames);


}

}
