#include "system.h"
#include "camera.h"
#include "object.h"
#include "mathutils.h"
#include "viszutils.h"
#include "frame.h"

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

    std::vector<Mat44_t> cameraPoses;
    std::vector<ThreeDDetection> landmarks;
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
        std::vector<ThreeDDetection> threeDDetections = oneFrame.Get3DDetections();
        for (ThreeDDetection oneDetection : threeDDetections){
            Eigen::Matrix<float, 2, Eigen::Dynamic> dstPoints = Eigen::Matrix<float, 2, Eigen::Dynamic>::Zero(2, oneDetection._vertices3DInCamera.cols());
            myStereoCamera.ProjectPoints(oneDetection._vertices3DInCamera, dstPoints);
            viszutils::Draw3DBoundingBox(dstPoints, display3DDetections);
        }
        std::filesystem::path debug3DDetectionPath = sStereoSequencePath;
        debug3DDetectionPath.append("debug3DDetection/").append(filename + ".png");
        cv::imwrite(debug3DDetectionPath.string() , display3DDetections);

        // // ransac based plane estimation
           // at the same time, ground plane based detection filtering

        // // input the correct detections to frameTracker, get pose return from frameTracker and print.

        if (filename == "000000"){
            cameraPoses.push_back(cameraInWorldTransform);
            for (ThreeDDetection oneDetection : threeDDetections){
                landmarks.push_back(oneDetection);
            }
        }
        else{
            // project 3d landmarks and 3d detections to current camera pose and find correspondences.
            size_t countLandmark = 0;
            std::vector<size_t> indicesCorrespondingDetecton;
            indicesCorrespondingDetecton.reserve(landmarks.size());
            size_t minOverlapAreaToReject = 400;
            cv::Mat displayLandmarks(myStereoCamera._rows, myStereoCamera._cols, CV_8UC1, cv::Scalar(0));
            cv::Mat displayDetections(myStereoCamera._rows, myStereoCamera._cols, CV_8UC1, cv::Scalar(0));
            for (ThreeDDetection landmark : landmarks){
                Eigen::MatrixXf transformedVertices = mathutils::TransformPoints<Eigen::MatrixXf>((nextFrameInCameraTransform * cameraInWorldTransform).inverse(), landmark._vertices3DInCamera);
                std::vector<cv::Point> points2DCV = mathutils::ProjectPoints3DToPoints2D(transformedVertices, myStereoCamera);
                cv::Mat landmarkPoseMask = mathutils::Draw2DHullMaskFrom2DPointsSet(points2DCV, myStereoCamera._rows, myStereoCamera._cols);
                int landmarkPoseMaskDataType = landmarkPoseMask.type();
                int displayLandmarksDataType = displayLandmarks.type();
                cv::bitwise_or(landmarkPoseMask, displayLandmarks, displayLandmarks);
                size_t countDetection = 0;
                int indexLargestOverlap = -1;
                int largestOverlapArea = 0;
                for (ThreeDDetection currentDetection : threeDDetections){
                    std::vector<cv::Point> points2DCV = mathutils::ProjectPoints3DToPoints2D(currentDetection._vertices3DInCamera, myStereoCamera);
                    cv::Mat currentDetectionPoseMask = mathutils::Draw2DHullMaskFrom2DPointsSet(points2DCV, myStereoCamera._rows, myStereoCamera._cols);
                    cv::Mat overlaps;
                    cv::bitwise_and(landmarkPoseMask, currentDetectionPoseMask, overlaps);
                    cv::Scalar sum = cv::sum(overlaps);
                    if (sum[0] > largestOverlapArea && sum[0] > minOverlapAreaToReject){
                        indexLargestOverlap = countDetection;
                    }
                    // TDO_LOG_DEBUG_FORMAT("Landmark No.%d, detection No.%d, overlapping area: %d", countLandmark % countDetection % sum[0]);
                    countDetection++;
                }
                if (indexLargestOverlap >= 0){
                    indicesCorrespondingDetecton.push_back(indexLargestOverlap);
                    std::vector<cv::Point> points2DCV = mathutils::ProjectPoints3DToPoints2D(threeDDetections[indexLargestOverlap]._vertices3DInCamera, myStereoCamera);
                    cv::Mat detectionPoseMask = mathutils::Draw2DHullMaskFrom2DPointsSet(points2DCV, myStereoCamera._rows, myStereoCamera._cols);
                    cv::bitwise_or(detectionPoseMask, displayDetections, displayDetections);
                }
                countLandmark++;
            }
            std::vector<cv::Mat> channels;
            cv::Mat zeroChannel(myStereoCamera._rows, myStereoCamera._cols, CV_8UC1, cv::Scalar(0));
            cv::Scalar sum = cv::sum(displayLandmarks);
            cv::Scalar sum2 = cv::sum(displayDetections);
            TDO_LOG_DEBUG("displayLandmarks sum: " << sum[0]);
            TDO_LOG_DEBUG("displayDetections sum: " << sum2[0]);
            channels.push_back(displayLandmarks * 255);
            channels.push_back(zeroChannel * 255);
            channels.push_back(displayDetections * 255);
            cv::Mat debugTracking;
            cv::merge(channels, debugTracking);
            std::filesystem::path debugTrackingPath = sStereoSequencePath;
            debugTrackingPath.append("debugTracking/").append(filename + ".png");
            cv::imwrite(debugTrackingPath.string() , debugTracking);
            // 3D object points in world coordinates
            std::vector<cv::Point3f> objectPoints;
            // Populate objectPoints with the corresponding 3D coordinates of the object
            // 2D image points in image coordinates
            std::vector<cv::Point2f> imagePoints;
            // Populate imagePoints with the corresponding 2D coordinates of the object in the image
            size_t indexLandmark = 0;
            for (size_t indexCorrespondingDetection : indicesCorrespondingDetecton){
                cv::Point3f point3D(
                    landmarks[indexLandmark]._objectInCameraTransform(0, 3),
                    landmarks[indexLandmark]._objectInCameraTransform(1, 3),
                    landmarks[indexLandmark]._objectInCameraTransform(2, 3)
                );
                objectPoints.push_back(point3D);
                cv::Point2f point2D(
                    (*matchedLeftCamDetections[indexCorrespondingDetection])._centerX,
                    (*matchedLeftCamDetections[indexCorrespondingDetection])._centerY
                );
                imagePoints.push_back(point2D);
                indexLandmark++;
            }
            // Camera intrinsic matrix (3x3)
            cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
            cameraMatrix.at<double>(0, 0) = static_cast<double>(myStereoCamera._kk(0, 0));
            cameraMatrix.at<double>(0, 2) = static_cast<double>(myStereoCamera._kk(0, 2));
            cameraMatrix.at<double>(1, 1) = static_cast<double>(myStereoCamera._kk(1, 1));
            cameraMatrix.at<double>(1, 2) = static_cast<double>(myStereoCamera._kk(1, 2));
            // Set the appropriate values for the cameraMatrix
            // Distortion coefficients
            cv::Mat distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
            // Set the appropriate values for the distCoeffs
            // Rotation vector and translation vector
            cv::Mat rvec, tvec;
            // Estimate camera pose using PnP
            cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);
            cv::Mat rotationMatrix;
            cv::Rodrigues(rvec, rotationMatrix);
            TDO_LOG_DEBUG("rvec: " << rvec << ", tvec: " << tvec);
            Mat44_t worldInCurrentCamera = Eigen::Matrix4f::Identity();
            for (int i=0; i < 3; i++){
                for (int j=0; j<3; j++){
                    worldInCurrentCamera(i, j) = rotationMatrix.at<double>(i, j);
                }
            }
            worldInCurrentCamera(0, 3) = tvec.at<double>(0);
            worldInCurrentCamera(1, 3) = tvec.at<double>(1);
            worldInCurrentCamera(2, 3) = tvec.at<double>(2);
            Mat44_t currentCameraInWorld = worldInCurrentCamera.inverse();
            TDO_LOG_DEBUG("currentCameraInWorld: \n" << currentCameraInWorld);
            if (currentCameraInWorld(2, 3) < -1.0 || currentCameraInWorld(2, 3) > 1.0){
                // track failed
                nextFrameInCameraTransform = Eigen::Matrix4f::Identity();
                // not updating cameraInWorld
            }
            else{
                nextFrameInCameraTransform(0, 3) = currentCameraInWorld(0, 3) - cameraInWorldTransform(0, 3);
                nextFrameInCameraTransform(1, 3) = currentCameraInWorld(1, 3) - cameraInWorldTransform(1, 3);
                nextFrameInCameraTransform(2, 3) = currentCameraInWorld(2, 3) - cameraInWorldTransform(2, 3);
                cameraInWorldTransform(0, 3) = currentCameraInWorld(0, 3);
                cameraInWorldTransform(1, 3) = currentCameraInWorld(1, 3);
                cameraInWorldTransform(2, 3) = currentCameraInWorld(2, 3);
            }

        }
        TDO_LOG_DEBUG("------------- End of one frame ------------");
        Mat44_t cameraInRealWorld = worldInRealWorld * cameraInWorldTransform;
        Eigen::Quaternionf myQuaternion(cameraInRealWorld.block<3, 3>(0, 0));
        outputFile << std::to_string(frameCount) << " " << cameraInRealWorld(0, 3) << " " << cameraInRealWorld(1, 3) << " " << cameraInRealWorld(2, 3) << " " << myQuaternion.x() << " " << myQuaternion.y() << " " << myQuaternion.z() << " " << myQuaternion.w() << std::endl;
        TDO_LOG_DEBUG("cameraInRealWorld: \n" << cameraInRealWorld);
        frameCount++;

        if (filename == "000001"){
            break;
        }
    }
    outputFile.close();
}

}
