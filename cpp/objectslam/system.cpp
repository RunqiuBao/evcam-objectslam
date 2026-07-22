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
    _sStereoSequencePathForDebug = debugPath;
}

// load the input image of one side of the stereo pair from concentrated/<side>/; black canvas if not on disk.
static cv::Mat LoadInputImageForDebugView(
    const std::string& sequencePath,
    const std::string& side,
    const std::string& timestamp,
    const unsigned int cols,
    const unsigned int rows
)
{
    std::filesystem::path imagePath = sequencePath;
    imagePath.append("concentrated/" + side + "/").concat(timestamp + ".png");
    if (std::filesystem::exists(imagePath)){
        cv::Mat image = cv::imread(imagePath.string(), cv::IMREAD_COLOR);
        if (!image.empty()){
            return image;
        }
    }
    return cv::Mat::zeros(rows, cols, CV_8UC3);
}

static void DrawDetectionsForDebugView(const std::vector<TwoDBoundingBox>& detections, cv::Mat& displayImage){
    for (const TwoDBoundingBox& oneDetection : detections){
        cv::Rect bbox(
            cvRound(oneDetection._centerX - oneDetection._bWidth / 2.f),
            cvRound(oneDetection._centerY - oneDetection._bHeight / 2.f),
            cvRound(oneDetection._bWidth),
            cvRound(oneDetection._bHeight)
        );
        cv::rectangle(displayImage, bbox, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        char scoreText[16];
        std::snprintf(scoreText, sizeof(scoreText), "%.2f", oneDetection._detectionScore);
        cv::putText(displayImage, scoreText, cv::Point(bbox.x, bbox.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    }
}

void SLAMSystem::UpdateDebugView(
    const std::shared_ptr<Frame>& pFrame,
    const bool isTrackingSuccess,
    const Mat44_t& framePoseInWorld,
    const std::string& keyframeInsertReason,
    const std::string& trackingMethod
)
{
    cv::Mat leftImage = LoadInputImageForDebugView(_sStereoSequencePathForDebug, "left", pFrame->_timestamp, _camera->_cols, _camera->_rows);
    cv::Mat rightImage = LoadInputImageForDebugView(_sStereoSequencePathForDebug, "right", pFrame->_timestamp, _camera->_cols, _camera->_rows);

    DrawDetectionsForDebugView(pFrame->_leftCamDetections, leftImage);
    DrawDetectionsForDebugView(pFrame->_rightCamDetections, rightImage);

    if (!isTrackingSuccess){
        cv::putText(leftImage, "TRACKING FAILURE", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }
    else{
        // absolute yaw of the frame in the world (poses are yaw-only: R(0,0)=cos, R(0,2)=sin).
        const float yawDeg = std::atan2(framePoseInWorld(0, 2), framePoseInWorld(0, 0)) * 180.f / M_PI;
        char positionText[96];
        std::snprintf(positionText, sizeof(positionText), "cam position: (%.2f, %.2f, %.2f), yaw: %.1f deg", framePoseInWorld(0, 3), framePoseInWorld(1, 3), framePoseInWorld(2, 3), yawDeg);
        cv::putText(leftImage, positionText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
    }
    cv::putText(leftImage, pFrame->_timestamp, cv::Point(10, leftImage.rows - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

    // ref keyframe id and the index of this frame within the ref keyframe.
    if (_frameTracker && _frameTracker->_pRefKeyframe){
        std::shared_ptr<KeyFrame> pRefKeyframe = _frameTracker->_pRefKeyframe;
        int indexInKeyframe = -1;
        if (pRefKeyframe->_vFrames_ids.count(pFrame) > 0){
            indexInKeyframe = 0;
            for (const auto& frame_id : pRefKeyframe->_vFrames_ids){
                if (frame_id.second < pFrame->_frameID){
                    indexInKeyframe++;
                }
            }
        }
        char keyframeText[64];
        std::snprintf(keyframeText, sizeof(keyframeText), "ref keyframe: %u, frame index: %d", pRefKeyframe->_keyFrameID, indexInKeyframe);
        cv::putText(leftImage, keyframeText, cv::Point(10, 55), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
    }

    // notify a keyframe insertion on this frame, with the reason.
    if (!keyframeInsertReason.empty()){
        cv::putText(leftImage, "KEYFRAME INSERTED: " + keyframeInsertReason, cv::Point(10, 80), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 255), 2, cv::LINE_AA);
    }

    // the tracking method that succeeded on this frame.
    if (isTrackingSuccess && !trackingMethod.empty()){
        cv::putText(leftImage, "tracker: " + trackingMethod, cv::Point(10, 105), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    }

    // object instance (landmark) ids below the bottom of the left bbox of each matched detection.
    if (pFrame->_pRefKeyframe){
        const std::vector<int>& detectionIDsOfRefObjects = pFrame->_detectionIDsOfCorrespondingRefObjects;
        for (size_t indexDetection = 0; indexDetection < pFrame->_matchedLeftCamDetections.size(); indexDetection++){
            // resolve the detection to its ref object, then to the object instance (landmark).
            // Note: a detection without an instance (e.g. a new object before its creation) gets no label;
            // the label appears once the object instance has been created.
            std::string instanceText;
            for (size_t indexRefObject = 0; indexRefObject < detectionIDsOfRefObjects.size(); indexRefObject++){
                if (detectionIDsOfRefObjects[indexRefObject] == static_cast<int>(indexDetection)){
                    std::shared_ptr<LandMark> pLandmark = pFrame->_pRefKeyframe->GetLandmarkByRefObjIndex(indexRefObject);
                    if (pLandmark){
                        instanceText = "obj " + std::to_string(pLandmark->_landmarkID);
                    }
                    break;
                }
            }
            if (instanceText.empty()){
                continue;
            }
            const TwoDBoundingBox& bbox = *pFrame->_matchedLeftCamDetections[indexDetection];
            cv::putText(leftImage, instanceText, cv::Point(cvRound(bbox._centerX - bbox._bWidth / 2.f), cvRound(bbox._centerY + bbox._bHeight / 2.f) + 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
        }
    }

    if (leftImage.size() != rightImage.size()){
        cv::resize(rightImage, rightImage, leftImage.size());
    }
    cv::Mat debugViewImage;
    cv::hconcat(leftImage, rightImage, debugViewImage);
    // separator line between the left and right image.
    cv::line(debugViewImage, cv::Point(leftImage.cols, 0), cv::Point(leftImage.cols, debugViewImage.rows - 1), cv::Scalar(255, 255, 255), 2);
    _pMapDb->SetDebugViewImage(debugViewImage);
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

void SaveOptimizedTraj(const std::string datasetRoot, std::vector<std::shared_ptr<Frame>> pFrameStack, const std::vector<Mat44Unaligned_t>& smoothedTrajectoryInWorld, const Mat44_t worldInReadworldTransform){
    std::filesystem::path trajFilePath = datasetRoot;
    trajFilePath.append("cameraTrackOptimized.txt");
    std::filesystem::path trajMaskFilePath = datasetRoot;
    trajMaskFilePath.append("cameraTrackOptimized_mask.txt");
    std::ofstream trajFile(trajFilePath.string());
    std::ofstream trajMaskFile(trajMaskFilePath.string());

    size_t frameCount = 0;
    for (auto pFrame : pFrameStack) {
        if (pFrame->_isTracked) {
            // save the smoothed trajectory (one smoothed world pose recorded per frame).
            Mat44_t frameInWorld = (frameCount < smoothedTrajectoryInWorld.size())
                ? smoothedTrajectoryInWorld[frameCount]
                : pFrame->_pRefKeyframe->GetKeyframePoseInWorld() * pFrame->GetPose();
            Mat44_t frameInRealWorld = worldInReadworldTransform * frameInWorld;
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
    _pMapDb->SetLatestFrameID(pOneFrame->_frameID);

    TDO_LOG_DEBUG_FORMAT("length of sDetection: %d", sDetections.size());
    if (sDetections.size() == 0){
        TDO_LOG_DEBUG("No detection in this frame. Early return with previous frame pose.");
        if (_frameTracker){
            _frameTracker->UpdateMOT(*pOneFrame);  // Note: advance the MOT (lost-track aging) even without detections.
        }
        Mat44_t frameInWorld = Eigen::Matrix4f::Identity();
        if (_pFrameStack.size() > 0) {
            frameInWorld = _pFrameStack.back()->GetPose();
            if (_pKeyFrameStack.size() > 0) {
                TDO_LOG_DEBUG("entered because _pKeyFrameStack.size() == " << _pKeyFrameStack.size());
                frameInWorld = _frameTracker->_pRefKeyframe->GetKeyframePoseInWorld() * frameInWorld;
            }
        }
        // no measurement this frame; the smoother coasts on its prediction.
        const Mat44_t smoothedFrameInWorld = _trajectorySmoother.Smooth(frameInWorld, false);
        _smoothedTrajectoryInWorld.push_back(smoothedFrameInWorld);
        if (isDebug){
            UpdateDebugView(pOneFrame, true, smoothedFrameInWorld, "", "");
        }
        return smoothedFrameInWorld;
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
    _frameTracker->UpdateMOT(*pOneFrame);  // Note: assigns BoT-SORT track ids to this frame's detections.
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
    bool isSuccess = true;  // Note: the (re)initialization branch always succeeds.
    std::string keyframeInsertReason;
    std::string trackingMethod;
    if (!_frameTracker->GetTrackerStatus()){
        keyframeInsertReason = "initialization";
        trackingMethod = "initialization";
        _maxNumDetectionsInHistory = std::max(_maxNumDetectionsInHistory, pOneFrame->_threeDDetections.size());
        pOneFrame->SetPose(Eigen::Matrix4f::Identity());
        pOneFrame->_isTracked = true;
        TDO_LOG_INFO("keyframe insert! frame number: " << timestamp);
        std::shared_ptr<KeyFrame> pOneKeyframe = std::make_shared<KeyFrame>(pOneFrame, Eigen::Matrix4f::Identity(), _camera);  // Note: allocated on heap. will not disappear due to out of scope.
        _frameTracker->_pRefKeyframe = pOneKeyframe;
        _frameTracker->SetTrackerStatus(true);
        _pKeyFrameStack.push_back(pOneKeyframe);
        _pMapDb->AddKeyFrame(pOneKeyframe);
        pOneFrame->SetDetectionsAsRefObjects();
        for (size_t indexDetection = 0; indexDetection < pOneFrame->_trackIDsOfDetections.size(); indexDetection++){
            if (pOneFrame->_trackIDsOfDetections[indexDetection] >= 0){
                pOneKeyframe->_trackIDToRefObjectIndex[pOneFrame->_trackIDsOfDetections[indexDetection]] = indexDetection;
            }
        }
        _frameTracker->CreateNewLandmarks(pOneKeyframe, _pMapDb, isDebug);
    }
    else{
        Mat44_t nextFrameInCameraTransformBackup = nextFrameInCameraTransform;  // Note: backup in case first track fails and nextFrameInCameraTransform will be set to identity,
        isSuccess = _frameTracker->DoSimple3DoFLeastSqauresTrack(*pOneFrame, (*_pFrameStack.back()), nextFrameInCameraTransform, isDebug);
        if (isSuccess){
            trackingMethod = "Simple3DoFLeastSqaures";
        }

        if (!isSuccess){
            isSuccess = _frameTracker->DoMotionBasedTrackDirect(*pOneFrame, (*_pFrameStack.back()), nextFrameInCameraTransform, isDebug);
            if (isSuccess){
                trackingMethod = "MotionBasedDirect";
            }
        }

        // if (!isSuccess){
        //     // TODO: project all ref objects to previous frame then track.
        //     bool isSuccess = _tracker.Do2DTrackingBasedTrack(*pOneFrame, (*_pFrameStack.back()), nextFrameInCameraTransform, isDebug);
        // }

        if (!isSuccess){
            isSuccess = _frameTracker->DoDenseAlignmentBasedTrackDirect(*pOneFrame, (*_pFrameStack.back()), isDebug);
            if (isSuccess){
                trackingMethod = "DenseAlignDirect";
                nextFrameInCameraTransform = (*_pFrameStack.back()).GetPose().inverse() * pOneFrame->GetPose();
            }
        }

        if (!isSuccess){
            // try relocalize from map
            isSuccess = _frameTracker->DoRelocalizeFromMap(*pOneFrame, (*_pFrameStack.back()), _pMapDb, nextFrameInCameraTransform, isDebug);
            if (isSuccess){
                trackingMethod = "Relocalize";
            }
            else{
                TDO_LOG_DEBUG("relocalization also failed.");
            }
        }
        
        // Note: for keyframe, all the left & right detections need to have confidences larger than threshold.
        const float keyframeDetectionConfidenceThreshold = 0.8;
        const bool confidencesIsOkay = CheckDetectionsConfidences(pOneFrame->_threeDDetections, keyframeDetectionConfidenceThreshold);
        // insert new key frame when a new object occurred, or when the tracked objects' bbox sizes changed by 30% on average w.r.t. the ref keyframe.
        const float keyframeAverageSizeChangeThreshold = 0.3f;
        bool hasNewObject = false;
        float averageSizeChange = 0.f;
        if (isSuccess){
            const std::vector<int>& detectionIDsOfRefObjects = pOneFrame->_detectionIDsOfCorrespondingRefObjects;
            const std::vector<std::shared_ptr<RefObject>>& refObjects = _frameTracker->_pRefKeyframe->_refObjects;
            std::vector<bool> isDetectionMatched(pOneFrame->_threeDDetections.size(), false);
            float sumSizeChange = 0.f;
            size_t countTrackedObjects = 0;
            for (size_t indexRefObject = 0; indexRefObject < std::min(detectionIDsOfRefObjects.size(), refObjects.size()); indexRefObject++){
                const int indexDetection = detectionIDsOfRefObjects[indexRefObject];
                if (indexDetection < 0 || indexDetection >= static_cast<int>(pOneFrame->_threeDDetections.size())){
                    continue;
                }
                isDetectionMatched[indexDetection] = true;
                const TwoDBoundingBox& refBbox = *refObjects[indexRefObject]->_detection._pLeftBbox;
                const TwoDBoundingBox& currBbox = *pOneFrame->_matchedLeftCamDetections[indexDetection];
                const float refArea = refBbox._bWidth * refBbox._bHeight;
                const float currArea = currBbox._bWidth * currBbox._bHeight;
                if (refArea > 0.f){
                    sumSizeChange += std::abs(currArea - refArea) / refArea;
                    countTrackedObjects++;
                }
            }
            averageSizeChange = (countTrackedObjects > 0)? sumSizeChange / countTrackedObjects : 0.f;
            // a new object: a detection that found no matched object instance in any frame until the ref keyframe.
            // Note: require good confidence on the new detection itself, so that spurious detections do not spawn objects.
            for (size_t indexDetection = 0; indexDetection < isDetectionMatched.size(); indexDetection++){
                if (isDetectionMatched[indexDetection]){
                    continue;
                }
                const ThreeDDetection& newDetection = pOneFrame->_threeDDetections[indexDetection];
                const float newObjectConfidenceThreshold = 0.5f;
                // Note: require a confirmed BoT-SORT track (matched in >= 2 frames) so that a spurious
                // single-frame detection cannot force a keyframe.
                const bool isConfirmedTrack = indexDetection < pOneFrame->_trackHitsOfDetections.size()
                    && pOneFrame->_trackHitsOfDetections[indexDetection] >= 2;
                if (isConfirmedTrack
                    && newDetection._pLeftBbox->_detectionScore >= newObjectConfidenceThreshold
                    && newDetection._pRightBbox->_detectionScore >= newObjectConfidenceThreshold){
                    hasNewObject = true;
                    TDO_LOG_DEBUG_FORMAT("detection %d is a new object (no instance matched until the ref keyframe).", indexDetection);
                    break;
                }
            }
        }
        const bool sizeChangedSignificantly = averageSizeChange >= keyframeAverageSizeChangeThreshold;
        // a confident new object always forces a new keyframe hosting it (its ref object and landmark are created
        // via the KeyFrame constructor and CreateNewLandmarks below); the size-change trigger additionally requires
        // all detections of the frame to be confident.
        // insert a keyframe when the detections reach the history maximum while the current ref keyframe hosts
        // fewer objects (e.g. it was created during a partial occlusion). The second clause stops the trigger
        // from re-firing on every frame that merely stays at the maximum.
        const bool hitsHistoryMax = pOneFrame->_threeDDetections.size() >= _maxNumDetectionsInHistory
            && pOneFrame->_threeDDetections.size() > _frameTracker->_pRefKeyframe->_refObjects.size();
        _maxNumDetectionsInHistory = std::max(_maxNumDetectionsInHistory, pOneFrame->_threeDDetections.size());
        // Note: a frame containing objects that cannot be explained by the ref keyframe (hasNewObject) must
        // always create a new keyframe to hold the novel objects.
        if (isSuccess && (hasNewObject || hitsHistoryMax || (confidencesIsOkay && sizeChangedSignificantly))){
            TDO_LOG_DEBUG_FORMAT("keyframe condition met: hasNewObject=%d, averageSizeChange=%f", static_cast<int>(hasNewObject) % averageSizeChange);
            if (hasNewObject){
                keyframeInsertReason = "new object";
            }
            if (hitsHistoryMax){
                keyframeInsertReason += (keyframeInsertReason.empty()? "" : " + ") + std::string("detections at history max");
            }
            if (sizeChangedSignificantly){
                char sizeChangeText[48];
                std::snprintf(sizeChangeText, sizeof(sizeChangeText), "size change %.0f%%", averageSizeChange * 100.f);
                keyframeInsertReason += (keyframeInsertReason.empty()? "" : " + ") + std::string(sizeChangeText);
            }
            TDO_LOG_DEBUG_FORMAT("Last Keyframe(%d) contains %d frames.", _pKeyFrameStack.back()->_keyFrameID % _pKeyFrameStack.back()->_vFrames_ids.size());
            // inherit object instances through the tracking association: detection j was tracked from old ref
            // object refIdx, whose landmark is the same physical object. Prevents duplicated instances when
            // the geometric re-association in CreateNewLandmarks fails (e.g. after a large size/depth change).
            std::shared_ptr<KeyFrame> pOldRefKeyframe = _frameTracker->_pRefKeyframe;
            const std::vector<int> detectionIDsOfOldRefObjects = pOneFrame->_detectionIDsOfCorrespondingRefObjects;  // Note: copy; overwritten by SetDetectionsAsRefObjects below.
            std::vector<std::shared_ptr<LandMark>> inheritedLandmarks(pOneFrame->_threeDDetections.size(), nullptr);
            for (size_t indexRefObject = 0; indexRefObject < detectionIDsOfOldRefObjects.size(); indexRefObject++){
                const int indexDetection = detectionIDsOfOldRefObjects[indexRefObject];
                if (indexDetection < 0 || indexDetection >= static_cast<int>(inheritedLandmarks.size())){
                    continue;
                }
                std::shared_ptr<LandMark> pTheLandmark = pOldRefKeyframe->GetLandmarkByRefObjIndex(indexRefObject);
                if (pTheLandmark && !pTheLandmark->IsToDelete()){
                    inheritedLandmarks[indexDetection] = pTheLandmark;
                }
            }
            std::shared_ptr<KeyFrame> pOneKeyframe = std::make_shared<KeyFrame>(pOneFrame, _frameTracker->_pRefKeyframe->GetKeyframePoseInWorld(), _camera);
            _frameTracker->_pRefKeyframe = pOneKeyframe;
            _pMapDb->AddKeyFrame(pOneKeyframe);
            _pKeyFrameStack.push_back(pOneKeyframe);
            pOneFrame->SetDetectionsAsRefObjects();
            for (size_t indexDetection = 0; indexDetection < pOneFrame->_trackIDsOfDetections.size(); indexDetection++){
                if (pOneFrame->_trackIDsOfDetections[indexDetection] >= 0){
                    pOneKeyframe->_trackIDToRefObjectIndex[pOneFrame->_trackIDsOfDetections[indexDetection]] = indexDetection;
                }
            }
            pOneFrame->SetPose(Eigen::Matrix4f::Identity());
            TDO_LOG_INFO("keyframe insert! frame number: " << timestamp);
            TDO_LOG_INFO("keyframe in world pose: " << pOneKeyframe->GetKeyframePoseInWorld());
            _frameTracker->CreateNewLandmarks(pOneKeyframe, _pMapDb, isDebug, inheritedLandmarks);
            // if (_pKeyFrameStack.size() > 5 && _pKeyFrameStack[_pKeyFrameStack.size() - 5]->_bContainNewLandmarks) {
            //     TDO_LOG_DEBUG_FORMAT("Entered landmark pruning at keyframe(%d).", _pKeyFrameStack[_pKeyFrameStack.size() - 5]->_keyFrameID);
            //     _pMapper->SchedulePruneLandmarksTask(_pKeyFrameStack[_pKeyFrameStack.size() - 5]);
            // }
            if (_pKeyFrameStack.size() % 3 == 2) {  // Note: every 3 keyframes, prune landmarks.
                _pMapper->SchedulePruneLandmarksTask();
            }
            // start BA
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
    const Mat44_t frameInWorld = _frameTracker->_pRefKeyframe->GetKeyframePoseInWorld() * pOneFrame->GetPose();
    // smooth the rough estimated pose returned by the trackers (online kalman, see trajectorysmoother.h).
    // Note: internal frame poses stay raw; only the published/returned pose is smoothed.
    const Mat44_t smoothedFrameInWorld = _trajectorySmoother.Smooth(frameInWorld, isSuccess);
    _smoothedTrajectoryInWorld.push_back(smoothedFrameInWorld);
    _pMapDb->SetCurrentFramePoseInWorld(smoothedFrameInWorld);

    // stamp the observation age of the landmarks matched in this frame (for age-based pruning).
    if (isSuccess){
        const std::vector<int>& detectionIDsOfRefObjects = pOneFrame->_detectionIDsOfCorrespondingRefObjects;
        for (size_t indexRefObject = 0; indexRefObject < detectionIDsOfRefObjects.size(); indexRefObject++){
            if (detectionIDsOfRefObjects[indexRefObject] < 0){
                continue;
            }
            std::shared_ptr<LandMark> pTheLandmark = _frameTracker->_pRefKeyframe->GetLandmarkByRefObjIndex(indexRefObject);
            if (pTheLandmark && !pTheLandmark->IsToDelete()){
                pTheLandmark->SetLastObservedFrameID(pOneFrame->_frameID);
            }
        }
    }

    // mark failed frames; at the next success, linearly interpolate their poses between the bracketing successes.
    if (!isSuccess){
        _framesPendingInterpolation.push_back(pOneFrame);
    }
    else{
        if (!_framesPendingInterpolation.empty()){
            const Vec3_t startTranslation = _lastSuccessfulFrameWorldPose.col(3).head<3>();
            const Vec3_t endTranslation = frameInWorld.col(3).head<3>();
            const Eigen::Quaternionf startRotation(_lastSuccessfulFrameWorldPose.block<3, 3>(0, 0));
            const Eigen::Quaternionf endRotation(frameInWorld.block<3, 3>(0, 0));
            const size_t numPending = _framesPendingInterpolation.size();
            for (size_t indexPending = 0; indexPending < numPending; indexPending++){
                std::shared_ptr<Frame> pPendingFrame = _framesPendingInterpolation[indexPending];
                if (!pPendingFrame->_pRefKeyframe){
                    continue;
                }
                const float fraction = static_cast<float>(indexPending + 1) / static_cast<float>(numPending + 1);
                Mat44_t interpolatedWorldPose = Eigen::Matrix4f::Identity();
                interpolatedWorldPose.block<3, 3>(0, 0) = startRotation.slerp(fraction, endRotation).toRotationMatrix();
                interpolatedWorldPose.col(3).head<3>() = (1.f - fraction) * startTranslation + fraction * endTranslation;
                pPendingFrame->SetPose(pPendingFrame->_pRefKeyframe->GetKeyframePoseInWorld().inverse() * interpolatedWorldPose);
                pPendingFrame->_isTracked = true;
                pPendingFrame->_isPoseInterpolated = true;
            }
            TDO_LOG_DEBUG_FORMAT("interpolated %d failed frames between the last two successful trackings.", numPending);
            _framesPendingInterpolation.clear();
        }
        _lastSuccessfulFrameWorldPose = frameInWorld;
    }

    if (isDebug){
        UpdateDebugView(pOneFrame, isSuccess, smoothedFrameInWorld, keyframeInsertReason, trackingMethod);
    }
    return smoothedFrameInWorld;
}

}
