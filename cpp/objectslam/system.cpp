#include "system.h"

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

    if (_jsonConfigNode.HasMember("tracker")) {
        const auto& t = _jsonConfigNode["tracker"];
        if (t.HasMember("minIoUToReject"))            _trackerParams.minIoUToReject            = t["minIoUToReject"].GetFloat();
        if (t.HasMember("minIoUToRejectForCloseObject")) _trackerParams.minIoUToRejectForCloseObject = t["minIoUToRejectForCloseObject"].GetFloat();
        if (t.HasMember("distanceCloseEnough"))       _trackerParams.distanceCloseEnough       = t["distanceCloseEnough"].GetFloat();
        if (t.HasMember("maxPoseError"))              _trackerParams.maxPoseError              = t["maxPoseError"].GetFloat();
        if (t.HasMember("maxPoseErrorInX"))           _trackerParams.maxPoseErrorInX           = t["maxPoseErrorInX"].GetFloat();
        if (t.HasMember("maxPoseErrorBA"))            _trackerParams.maxPoseErrorBA            = t["maxPoseErrorBA"].GetFloat();
        if (t.HasMember("maxlandmarkErrorBA"))        _trackerParams.maxlandmarkErrorBA        = t["maxlandmarkErrorBA"].GetFloat();
        if (t.HasMember("maxRotationAngleDeg"))       _trackerParams.maxRotationAngleDeg       = t["maxRotationAngleDeg"].GetFloat();
    }
}

SystemConfig::SystemConfig(const SystemConfig& config){
    _configFilePath = config._configFilePath;
    _jsonConfigNode.CopyFrom(config._jsonConfigNode, _jsonConfigNode.GetAllocator());
}

SLAMSystem::SLAMSystem(const std::shared_ptr<SystemConfig>& cfg)
:_cfg(cfg)
{
    _pMapDb = std::make_shared<MapDataBase>();
    _pMapper = std::make_unique<SemanticMapper>(_pMapDb, cfg->_trackerParams.maxPoseErrorBA);
}

void SLAMSystem::Startup() {
    TDO_LOG_DEBUG("Startup SLAM system.");
    _pMapperThread  = std::unique_ptr<std::thread>(new std::thread(&SemanticMapper::Run, _pMapper.get()));
}

void SLAMSystem::InitializeCameraAndTracker(
    const unsigned int cameraID,
    const unsigned int cols,
    const unsigned int rows,
    const Eigen::Matrix3f kk,
    const float baseline,
    const std::string& debugPath
)
{
    _camera = std::make_shared<camera::CameraBase>(
        0,
        cols,
        rows,
        kk,
        baseline
    );
    TDO_LOG_DEBUG_FORMAT("myStereoCamera imageWidth: %d", cols);
    nextFrameInCameraTransform = Eigen::Matrix4f::Identity();

    _frameTracker = std::make_unique<FrameTracker>(_camera, _cfg->_trackerParams);
    _frameTracker->_sStereoSequencePathForDebug = debugPath;  // To dump debug images.
}

void LoadDetectionsWithFacet(
    const std::vector<std::string>& sDetections,
    std::vector<TwoDBoundingBox>& leftDetections,
    std::vector<TwoDBoundingBox>& rightDetections,
    std::shared_ptr<object::ObjectBase> pFacetObject    
){
    leftDetections.reserve(sDetections.size());
    rightDetections.reserve(sDetections.size());
    for (std::string sDetection : sDetections){
        std::vector<std::string> splitSDetection;
        boost::split(splitSDetection, sDetection, boost::is_any_of(" "));
        Vec2_t corner0(
            boost::lexical_cast<float>(splitSDetection[1].c_str()),
            boost::lexical_cast<float>(splitSDetection[2].c_str())
        );
        Vec2_t corner1(
            boost::lexical_cast<float>(splitSDetection[3].c_str()),
            boost::lexical_cast<float>(splitSDetection[4].c_str())
        );
        Vec2_t corner2(
            boost::lexical_cast<float>(splitSDetection[5].c_str()),
            boost::lexical_cast<float>(splitSDetection[6].c_str())
        );
        Vec2_t corner3(
            boost::lexical_cast<float>(splitSDetection[7].c_str()),
            boost::lexical_cast<float>(splitSDetection[8].c_str())
        );

        std::vector<Vec2_t> corners_left = {corner0, corner1, corner2, corner3};  // all the corners are in the order of clockwise, from topleft point. Assmuing rotation around z be less than 45 deg.
        // compute a bounding box for the corners
        float min_x = corners_left[0][0], min_y = corners_left[0][1];
        float max_x = corners_left[0][0], max_y = corners_left[0][1];
        for (const auto& corner : corners_left) {
            min_x = std::min(min_x, corner[0]);
            min_y = std::min(min_y, corner[1]);
            max_x = std::max(max_x, corner[0]);
            max_y = std::max(max_y, corner[1]);
        }

        // TODO: fix facet for new detection format
        // leftDetections.push_back(
        //     TwoDBoundingBox(
        //         (min_x + max_x) / 2,
        //         (min_y + max_y) / 2,
        //         max_x - min_x,
        //         max_y - min_y,
        //         pFacetObject, 
        //         1,
        //         corners_left,
        //         true
        //     )
        // );
        TDO_LOG_DEBUG_FORMAT("one 2d facet at (left): %f, %f", ((min_x + max_x) / 2) % ((min_y + max_y) / 2));

        Vec2_t corner0_r(
            boost::lexical_cast<float>(splitSDetection[9].c_str()),
            boost::lexical_cast<float>(splitSDetection[10].c_str())
        );
        Vec2_t corner1_r(
            boost::lexical_cast<float>(splitSDetection[11].c_str()),
            boost::lexical_cast<float>(splitSDetection[12].c_str())
        );
        Vec2_t corner2_r(
            boost::lexical_cast<float>(splitSDetection[13].c_str()),
            boost::lexical_cast<float>(splitSDetection[14].c_str())
        );
        Vec2_t corner3_r(
            boost::lexical_cast<float>(splitSDetection[15].c_str()),
            boost::lexical_cast<float>(splitSDetection[16].c_str())
        );
        std::vector<Vec2_t> corners_right = {corner0_r, corner1_r, corner2_r, corner3_r};  // all the corners are in the order of clockwise, from topleft point. Assmuing rotation around z be less than 45 deg.
        // compute a bounding box for the corners
        min_x = corners_right[0][0], min_y = corners_right[0][1];
        max_x = corners_right[0][0], max_y = corners_right[0][1];
        for (const auto& corner : corners_right) {
            min_x = std::min(min_x, corner[0]);
            min_y = std::min(min_y, corner[1]);
            max_x = std::max(max_x, corner[0]);
            max_y = std::max(max_y, corner[1]);
        }

        std::string sConfidence = splitSDetection[17];
        sConfidence.pop_back();
        float confidence = boost::lexical_cast<float>(sConfidence);

        // rightDetections.push_back(
        //     TwoDBoundingBox(
        //         (min_x + max_x) / 2,
        //         (min_y + max_y) / 2,
        //         max_x - min_x,
        //         max_y - min_y,
        //         pFacetObject, 
        //         confidence,
        //         corners_right,
        //         true
        //     )
        // );
    }

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
        float x_tl, y_tl, x_br, y_br, x_tl_r, y_tl_r, x_br_r, y_br_r, cx, cy, bWidth, bHeight, cx_r, cy_r, bWidth_r, bHeight_r, detectionScore;
        x_tl = boost::lexical_cast<float>(splitSDetection[0].c_str());
        y_tl = boost::lexical_cast<float>(splitSDetection[1].c_str());
        x_br = boost::lexical_cast<float>(splitSDetection[2].c_str());
        y_br = boost::lexical_cast<float>(splitSDetection[3].c_str());
        x_tl_r = boost::lexical_cast<float>(splitSDetection[4].c_str());
        y_tl_r = boost::lexical_cast<float>(splitSDetection[5].c_str());
        x_br_r = boost::lexical_cast<float>(splitSDetection[6].c_str());
        y_br_r = boost::lexical_cast<float>(splitSDetection[7].c_str());
        cx = (x_tl + x_br) / 2;
        cy = (y_tl + y_br) / 2;
        cx_r = (x_tl_r + x_br_r) / 2;
        cy_r = (y_tl_r + y_br_r) / 2;
        bWidth = x_br - x_tl;
        bHeight = y_br - y_tl;
        bWidth_r = x_br_r - x_tl_r;
        bHeight_r = y_br_r - y_tl_r;
        // Vec2_t keypt1(
        //     boost::lexical_cast<float>(splitSDetection[11].c_str()),
        //     boost::lexical_cast<float>(splitSDetection[12].c_str())
        // );
        Vec2_t keypt1(
            cx,
            y_tl
        );
        Vec2_t keypt2(
            cx,
            cy
        );
        std::vector<Vec2_t> keypts = {keypt1, keypt2};
        Vec2_t tl_corner(
            x_tl,
            y_tl
        );
        Vec2_t tr_corner(
            x_br,
            y_tl
        );
        Vec2_t br_corner(
            x_br,
            y_br
        );
        Vec2_t bl_corner(
            x_tl,
            y_br
        );
        std::vector<Vec2_t> vertices2D = {tl_corner, tr_corner, br_corner, bl_corner};  // from topleft corner in clockwise direction.
        detectionScore = boost::lexical_cast<float>(splitSDetection[9].c_str());

        leftDetections.push_back(TwoDBoundingBox(cx, cy, bWidth, bHeight, pColorcone, detectionScore, vertices2D, keypts, false));
        // Vec2_t keypt1_r(
        //     boost::lexical_cast<float>(splitSDetection[17].c_str()),
        //     boost::lexical_cast<float>(splitSDetection[12].c_str())
        // );
        Vec2_t keypt1_r(
            cx_r,
            y_tl_r
        );
        Vec2_t keypt2_r(
            cx_r,
            cy_r
        );
        std::vector<Vec2_t> keypts_r = {keypt1_r, keypt2_r};
        Vec2_t tl_corner_r(
            x_tl_r,
            y_tl
        );
        Vec2_t tr_corner_r(
            x_br_r,
            y_tl
        );
        Vec2_t br_corner_r(
            x_br_r,
            y_br
        );
        Vec2_t bl_corner_r(
            x_tl_r,
            y_br
        );
        std::vector<Vec2_t> vertices2D_r = {tl_corner_r, tr_corner_r, br_corner_r, bl_corner_r};
        rightDetections.push_back(TwoDBoundingBox(cx_r, cy, bWidth_r, bHeight, pColorcone, detectionScore, vertices2D_r, keypts_r, false));
        TDO_LOG_DEBUG_FORMAT("one 2d detection (confi %f): %f, %f", detectionScore % cx % cy);
    }
}

const bool CheckDetectionsConfidences(const std::vector<ThreeDDetection>& threeDDetections, const float confidenceThreshold) {
    int count = 0;
    for (const ThreeDDetection& detection : threeDDetections) {
        if (detection._pLeftBbox->_detectionScore < confidenceThreshold || detection._pRightBbox->_detectionScore < confidenceThreshold) {
            TDO_LOG_DEBUG_FORMAT("%d-th Detection with score %f is below threshold.", count % detection._detectionScore);
            return false;
        }
    }
    return true;
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

const Mat44_t SLAMSystem::UpdateOneFrame(
    const std::string& timestamp,
    const std::vector<std::string>& sDetections,
    std::shared_ptr<object::ObjectBase> pTheLandmarkObject,
    const bool isDebug
)
{
    int frameCount = static_cast<int>(_allFramesStack.size());

    TDO_LOG_CRITICAL_FORMAT("UpdateOneFrame called with timestamp: %s", timestamp);
    std::shared_ptr<Frame> pOneFrame = std::make_shared<Frame>(FrameType::Stereo, timestamp, _camera);
    _allFramesStack.push_back(pOneFrame);

    TDO_LOG_DEBUG_FORMAT("length of sDetection: %d", sDetections.size());
    if (sDetections.size() == 0){
        TDO_LOG_DEBUG("No detection in this frame. Early return with previous frame pose.");
        Mat44_t frameInWorld = Eigen::Matrix4f::Identity();
        if (_pFrameStack.size() > 0) {
            frameInWorld = _pFrameStack.back()->GetPose();
            if (_pKeyFrameStack.size() > 0) {
                TDO_LOG_DEBUG("entered because _pKeyFrameStack.size() == " << _pKeyFrameStack.size());
                frameInWorld = _frameTracker->_pRefKeyframe->GetKeyframePoseInWorld() * frameInWorld;
            }
        }
        return frameInWorld;
    }

    std::vector<TwoDBoundingBox> leftCamDetections, rightCamDetections;  // Note: load left and right detections and list correspondingly.
    LoadDetections(
        sDetections,
        leftCamDetections,
        rightCamDetections,
        _camera->_cols,
        _camera->_rows,
        pTheLandmarkObject
    );
    pOneFrame->SetDetectionsFromExternalSrc(std::move(leftCamDetections), std::move(rightCamDetections));
    std::vector<std::shared_ptr<TwoDBoundingBox>> matchedLeftCamDetections, matchedRightCamDetections;
    auto matchedDetections = pOneFrame->GetMatchedDetections();  // Note: return a tuple.
    matchedLeftCamDetections = std::get<0>(matchedDetections);
    matchedRightCamDetections = std::get<1>(matchedDetections);
    // if (isDebug){
    //     std::filesystem::path leftCamImagePath = sStereoSequencePath;
    //     leftCamImagePath.append("concentrated/left/").append(timestamp + ".png");
    //     cv::Mat display3DDetections = cv::imread(leftCamImagePath.string(), cv::IMREAD_COLOR);
    //     std::vector<ThreeDDetection> threeDDetections = pOneFrame->_threeDDetections;
    //     int countThreeDDetection = 0;
    //     for (ThreeDDetection oneDetection : threeDDetections){
    //         Eigen::Matrix<float, 2, Eigen::Dynamic> dstPoints(2, 2);
    //         Eigen::Matrix<float, 3, Eigen::Dynamic> threeDKeypoints(3, 2);
    //         threeDKeypoints << oneDetection._keypt1InRefFrame(0), oneDetection._objectCenterInRefFrame(0),
    //                            oneDetection._keypt1InRefFrame(1), oneDetection._objectCenterInRefFrame(1),
    //                            oneDetection._keypt1InRefFrame(2), oneDetection._objectCenterInRefFrame(2);
    //         myStereoCamera.ProjectPoints(threeDKeypoints, dstPoints);
    //         viszutils::Draw5DDetections(dstPoints, display3DDetections);

    //         countThreeDDetection++;
    //     }

    //     std::filesystem::path debug3DDetectionPath = sStereoSequencePath;
    //     debug3DDetectionPath.append("debug3DDetections/");
    //     if (!std::filesystem::exists(debug3DDetectionPath) && !std::filesystem::create_directory(debug3DDetectionPath)){
    //         TDO_LOG_ERROR_FORMAT("Failed to create the folder: %s", debug3DDetectionPath.string());
    //         throw std::runtime_error("OS Error");
    //     }
    //     debug3DDetectionPath.append(timestamp + ".png");
    //     cv::imwrite(debug3DDetectionPath.string() , display3DDetections);
    // }

    /* input the correct detections to frameTracker, get pose return from frameTracker and print. */
    if (!_frameTracker->GetTrackerStatus()){
        pOneFrame->SetPose(Eigen::Matrix4f::Identity());
        pOneFrame->_isTracked = true;
        TDO_LOG_INFO("keyframe insert! frame number: " << timestamp);
        std::shared_ptr<KeyFrame> pOneKeyframe = std::make_shared<KeyFrame>(pOneFrame, Eigen::Matrix4f::Identity(), _camera);  // Note: allocated on heap. will not disappear due to out of scope.
        _frameTracker->_pRefKeyframe = pOneKeyframe;
        _frameTracker->SetTrackerStatus(true);
        _pKeyFrameStack.push_back(pOneKeyframe);
        _pMapDb->AddKeyFrame(pOneKeyframe);
        pOneFrame->SetDetectionsAsRefObjects();
        _frameTracker->CreateNewLandmarks(pOneKeyframe, _pMapDb, isDebug);
    }
    else{
        Mat44_t nextFrameInCameraTransformBackup = nextFrameInCameraTransform;  // Note: backup in case first track fails and nextFrameInCameraTransform will be set to identity,
        bool isSuccess = _frameTracker->DoMotionBasedTrack(*pOneFrame, (*_pFrameStack.back()), nextFrameInCameraTransform, isDebug);

        // if (!isSuccess){
        //     // TODO: project all ref objects to previous frame then track.
        //     bool isSuccess = _tracker.Do2DTrackingBasedTrack(*pOneFrame, (*_pFrameStack.back()), nextFrameInCameraTransform, isDebug);
        // }

        if (!isSuccess){
            isSuccess = _frameTracker->DoDenseAlignmentBasedTrack(*pOneFrame, (*_pFrameStack.back()), isDebug);
            if (isSuccess){
                nextFrameInCameraTransform = (*_pFrameStack.back()).GetPose().inverse() * pOneFrame->GetPose();
            }
        }

        if (!isSuccess){
            // try relocalize from map
            isSuccess = _frameTracker->DoRelocalizeFromMap(*pOneFrame, (*_pFrameStack.back()), _pMapDb, nextFrameInCameraTransform, isDebug);
            if (!isSuccess){
                TDO_LOG_DEBUG("relocalization also failed.");
            }
        }
        
        // insert new key frame if detection increased than previous frame
        // if (isSuccess && (pOneFrame->_threeDDetections.size() > (*_pFrameStack.back())._threeDDetections.size())){
        // Note: for keyframe, all the left & right detections need to have confidences larger than threshold.
        const float keyframeDetectionConfidenceThreshold = 0.8;
        const bool confidencesIsOkay = CheckDetectionsConfidences(pOneFrame->_threeDDetections, keyframeDetectionConfidenceThreshold);
        if (isSuccess && confidencesIsOkay && pOneFrame->_threeDDetections.size() >= 3 && (pOneFrame->_threeDDetections.size() > (*_pFrameStack.back())._threeDDetections.size())){
            TDO_LOG_DEBUG_FORMAT("Last Keyframe(%d) contains %d frames.", _pKeyFrameStack.back()->_keyFrameID % _pKeyFrameStack.back()->_vFrames_ids.size());
            std::shared_ptr<KeyFrame> pOneKeyframe = std::make_shared<KeyFrame>(pOneFrame, _frameTracker->_pRefKeyframe->GetKeyframePoseInWorld(), _camera);
            _frameTracker->_pRefKeyframe = pOneKeyframe;
            _pMapDb->AddKeyFrame(pOneKeyframe);
            _pKeyFrameStack.push_back(pOneKeyframe);
            pOneFrame->SetDetectionsAsRefObjects();
            pOneFrame->SetPose(Eigen::Matrix4f::Identity());
            TDO_LOG_INFO("keyframe insert! frame number: " << timestamp);
            TDO_LOG_INFO("keyframe in world pose: " << pOneKeyframe->GetKeyframePoseInWorld());
            _frameTracker->CreateNewLandmarks(pOneKeyframe, _pMapDb, isDebug);
            // if (_pKeyFrameStack.size() > 5 && _pKeyFrameStack[_pKeyFrameStack.size() - 5]->_bContainNewLandmarks) {
            //     TDO_LOG_DEBUG_FORMAT("Entered landmark pruning at keyframe(%d).", _pKeyFrameStack[_pKeyFrameStack.size() - 5]->_keyFrameID);
            //     _pMapper->SchedulePruneLandmarksTask(_pKeyFrameStack[_pKeyFrameStack.size() - 5]);
            // }
            if (_pKeyFrameStack.size() % 3 == 2) {  // Note: every 3 keyframes, prune landmarks.
                _pMapper->SchedulePruneLandmarksTask();
            }
            // start BA
            // bool doBA = pOneFrame
            while (!_pMapper->PushKeyframeForBA(pOneKeyframe)){
                // Note: Force optimize every keyframe.
                std::this_thread::sleep_for(std::chrono::microseconds(static_cast<unsigned int>(0.05 * 1e6)));
            }
        }

    }
    TDO_LOG_INFO("------------- End of one frame ------------");

    TDO_LOG_DEBUG("keyframe pose in world: \n" << _frameTracker->_pRefKeyframe->GetKeyframePoseInWorld());
    TDO_LOG_DEBUG("frame pose in keyframe: \n" << pOneFrame->GetPose());
    pOneFrame->_pRefKeyframe = _frameTracker->_pRefKeyframe;
    _frameTracker->_pRefKeyframe->_vFrames_ids[pOneFrame] = pOneFrame->_frameID;
    _pFrameStack.push_back(pOneFrame);
    return _frameTracker->_pRefKeyframe->GetKeyframePoseInWorld() * pOneFrame->GetPose();
}

}
