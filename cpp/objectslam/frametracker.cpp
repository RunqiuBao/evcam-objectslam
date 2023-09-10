#include "mathutils.h"
#include "frametracker.h"
#include "mapdatabase.h"
#include "landmark.h"

#include <filesystem>
#include <algorithm>
#include <opencv2/tracking.hpp>
#include <opencv2/tracking/tracking_legacy.hpp>

#include <logging.h>
TDO_LOGGER("eventobjectslam.frametracker")

namespace eventobjectslam {

// bool FrameTracker::DoBruteForceMatchBasedTrack(
//     frame& currentFrame,
//     const frame& lastFrame,
//     const Mat44_t& velocity
// ) const{
// // TODO: 
// // 1. each frame contains object instances (plane optimized) it detected. but they do not contain object id.
// // 2. according to roughly estimated pose, project existing object in and find correspondences; create new objects if exist; use pnp update frame pose, return True.
// // 3. if more than 4 objects detected, and not in line, create key frame.
// // 4. if not enough detection or correspondences, return false.

// }

static void TrackWithPnP(
    const std::vector<cv::Point3f>& objectPoints,
    const std::vector<cv::Point2f>& imagePoints,
    const cv::Mat& cameraMatrix,
    const cv::Mat& distCoeffs,
    const float maxPoseError,  // Note: assume camera pose should be close to the origin point. if over this error, use ransacpnp instead.
    Mat44_t& currentFrameInRefKeyFrame
){
    // Rotation vector and translation vector
    cv::Mat rvec, tvec;
    cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, false, cv::SOLVEPNP_EPNP);
    TDO_LOG_DEBUG("rvec: " << rvec << ", tvec: " << tvec);
    if (std::sqrt(tvec.at<double>(0)*tvec.at<double>(0) + tvec.at<double>(1)*tvec.at<double>(1) + tvec.at<double>(2)*tvec.at<double>(2)) > maxPoseError){
        cv::solvePnPRansac(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, false, 20, 3.0);  // iterationsCount = 20, 	reprojectionError = 3.0
    }
    TDO_LOG_DEBUG("(ransac) rvec: " << rvec << ", tvec: " << tvec);
    cv::Mat rotationMatrix;
    cv::Rodrigues(rvec, rotationMatrix);
    Mat44_t refKeyFrameInCurrentFrame = Eigen::Matrix4f::Identity();
    for (int i=0; i < 3; i++){
        for (int j=0; j<3; j++){
            refKeyFrameInCurrentFrame(i, j) = rotationMatrix.at<double>(i, j);
        }
    }
    refKeyFrameInCurrentFrame(0, 3) = tvec.at<double>(0);
    refKeyFrameInCurrentFrame(1, 3) = tvec.at<double>(1);
    refKeyFrameInCurrentFrame(2, 3) = tvec.at<double>(2);
    currentFrameInRefKeyFrame = refKeyFrameInCurrentFrame.inverse();
}

bool FrameTracker::DoMotionBasedTrack(Frame& currentFrame, const Frame& lastFrame, Mat44_t& velocity) const{
    size_t minOverlapAreaToReject = 400;
    float maxPoseError = 0.5;

    std::vector<std::shared_ptr<RefObject>> refObjects = _pRefKeyframe->_refObjects;
    // project 3d refObjects and 3d detections to current camera pose and find correspondences.
    std::vector<int> indicesCorrespondingDetecton;
    indicesCorrespondingDetecton.reserve(refObjects.size());
    size_t countRefObject = 0;
    camera::CameraBase myStereoCamera = *_camera;
    cv::Mat displayRefObjects(myStereoCamera._rows, myStereoCamera._cols, CV_8UC1, cv::Scalar(0));
    cv::Mat displayDetections(myStereoCamera._rows, myStereoCamera._cols, CV_8UC1, cv::Scalar(0));
    for (std::shared_ptr<RefObject> refObject : refObjects){
        Eigen::MatrixXf transformedVertices = mathutils::TransformPoints<Eigen::MatrixXf>((velocity * lastFrame.GetPose()).inverse(), refObject->_detection._vertices3DInCamera);
        std::vector<cv::Point> points2DCV = mathutils::ProjectPoints3DToPoints2D(transformedVertices, myStereoCamera);
        cv::Mat refObjectPoseMask = mathutils::Draw2DHullMaskFrom2DPointsSet(points2DCV, myStereoCamera._rows, myStereoCamera._cols);
        int refObjectPoseMaskDataType = refObjectPoseMask.type();
        int displayRefObjectsDataType = displayRefObjects.type();
        cv::bitwise_or(refObjectPoseMask, displayRefObjects, displayRefObjects);
        size_t countDetection = 0;
        int indexLargestOverlap = -1;
        int largestOverlapArea = 0;
        for (ThreeDDetection currentDetection : currentFrame._threeDDetections){
            std::vector<cv::Point> points2DCV = mathutils::ProjectPoints3DToPoints2D(currentDetection._vertices3DInCamera, myStereoCamera);
            cv::Mat currentDetectionPoseMask = mathutils::Draw2DHullMaskFrom2DPointsSet(points2DCV, myStereoCamera._rows, myStereoCamera._cols);
            cv::Mat overlaps;
            cv::bitwise_and(refObjectPoseMask, currentDetectionPoseMask, overlaps);
            cv::Scalar sum = cv::sum(overlaps);
            if (sum[0] > largestOverlapArea && sum[0] > minOverlapAreaToReject){
                indexLargestOverlap = countDetection;
                largestOverlapArea = sum[0];
            }
            // TDO_LOG_DEBUG_FORMAT("RefObject No.%d, detection No.%d, overlapping area: %d", countRefObject % countDetection % sum[0]);
            countDetection++;
        }
        if (indexLargestOverlap >= 0){
            std::vector<cv::Point> points2DCV = mathutils::ProjectPoints3DToPoints2D(currentFrame._threeDDetections[indexLargestOverlap]._vertices3DInCamera, myStereoCamera);
            cv::Mat detectionPoseMask = mathutils::Draw2DHullMaskFrom2DPointsSet(points2DCV, myStereoCamera._rows, myStereoCamera._cols);
            cv::bitwise_or(detectionPoseMask, displayDetections, displayDetections);
            TDO_LOG_DEBUG("found corresponding detection for refObject " << std::to_string(countRefObject) << ", overlap area :" << largestOverlapArea);
        }
        indicesCorrespondingDetecton.push_back(indexLargestOverlap);

        countRefObject++;
    }
    std::vector<cv::Mat> channels;
    cv::Mat zeroChannel(myStereoCamera._rows, myStereoCamera._cols, CV_8UC1, cv::Scalar(0));
    cv::Scalar sum = cv::sum(displayRefObjects);
    cv::Scalar sum2 = cv::sum(displayDetections);
    TDO_LOG_DEBUG("displayRefObjects sum: " << sum[0]);
    TDO_LOG_DEBUG("displayDetections sum: " << sum2[0]);
    channels.push_back(displayRefObjects * 255);
    channels.push_back(displayDetections * 255);
    channels.push_back(zeroChannel * 255);
    cv::Mat debugTracking;
    cv::merge(channels, debugTracking);
    std::filesystem::path debugTrackingPath = _sStereoSequencePathForDebug;
    debugTrackingPath.append("debugTracking/");
    if (!std::filesystem::exists(debugTrackingPath) && !std::filesystem::create_directory(debugTrackingPath)){
        TDO_LOG_ERROR_FORMAT("Failed to create the folder: %s", debugTrackingPath.string());
        return false;
    }
    debugTrackingPath.append(mathutils::FillZeros(std::to_string(static_cast<int>(currentFrame._timestamp)), 6) + ".png");
    cv::imwrite(debugTrackingPath.string() , debugTracking);
    // 3D object points in world coordinates
    std::vector<cv::Point3f> objectPoints;
    // Populate objectPoints with the corresponding 3D coordinates of the object
    // 2D image points in image coordinates
    std::vector<cv::Point2f> imagePoints;
    // Populate imagePoints with the corresponding 2D coordinates of the object in the image
    size_t indexRefObject = 0;
    for (int indexCorrespondingDetection : indicesCorrespondingDetecton){
        if (indexCorrespondingDetection < 0){
            indexRefObject++;
            continue;
        }
        cv::Point3f point3D(
            refObjects[indexRefObject]->_detection._objectInCameraTransform(0, 3),
            refObjects[indexRefObject]->_detection._objectInCameraTransform(1, 3),
            refObjects[indexRefObject]->_detection._objectInCameraTransform(2, 3)
        );
        objectPoints.push_back(point3D);
        cv::Point2f point2D(
            (*currentFrame._matchedLeftCamDetections[indexCorrespondingDetection])._centerX,
            (*currentFrame._matchedLeftCamDetections[indexCorrespondingDetection])._centerY
        );
        imagePoints.push_back(point2D);
        indexRefObject++;
    }

    // Estimate camera pose using PnP
    if (objectPoints.size() < 4){
        // track failed
        velocity = Eigen::Matrix4f::Identity();
        currentFrame.SetPose(velocity * lastFrame.GetPose());
        TDO_LOG_DEBUG("track fail. not updating camera pose.");
        // not updating cameraInWorld
        return false;
    }
    else{
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

        Mat44_t currentFrameInRefKeyFrame;
        TrackWithPnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, maxPoseError, currentFrameInRefKeyFrame);
        TDO_LOG_DEBUG("currentCameraInRefKeyFrame: \n" << currentFrameInRefKeyFrame);
        velocity = lastFrame.GetPose().inverse() * currentFrameInRefKeyFrame;  // Note: think like there is a point in last frame, first transform it to keyframe then to current frame.
        if (
            velocity.block(0, 3, 3, 1).norm() > maxPoseError
        ){
            // track failed
            velocity = Eigen::Matrix4f::Identity();
            currentFrame.SetPose(velocity * lastFrame.GetPose());
            TDO_LOG_DEBUG("track fail. not updating camera pose.");
            // not updating cameraInWorld
            return false;
        }
        else{
            // nextFrameInCameraTransform(0, 3) = currentCameraInWorld(0, 3) - cameraInWorldTransform(0, 3);
            // nextFrameInCameraTransform(1, 3) = currentCameraInWorld(1, 3) - cameraInWorldTransform(1, 3);
            // nextFrameInCameraTransform(2, 3) = currentCameraInWorld(2, 3) - cameraInWorldTransform(2, 3);
            // cameraInWorldTransform(0, 3) = currentCameraInWorld(0, 3);
            // cameraInWorldTransform(1, 3) = currentCameraInWorld(1, 3);
            // cameraInWorldTransform(2, 3) = currentCameraInWorld(2, 3);
            currentFrame.SetPose(currentFrameInRefKeyFrame);
            currentFrame._isTracked = true;
            currentFrame._detectionIDsOfCorrespondingRefObjects = indicesCorrespondingDetecton;
            TDO_LOG_DEBUG("track succeeded. currentFrameInRefFrame:\n" << velocity);
            return true;
        }
    }
}

static void DrawNodes(
    cv::Mat& canvas,
    const std::vector<ThreeDDetection>& threeDDetections,
    camera::CameraBase& camera,
    const int nodeSize
){
    canvas = cv::Mat::zeros(camera._rows, camera._cols, CV_8U);
    for (ThreeDDetection oneDetection : threeDDetections){
        Eigen::MatrixXf detectionPoseCenter = Eigen::MatrixXf::Zero(3, 1);
        detectionPoseCenter.block(0, 0, 3, 1) = oneDetection._objectInCameraTransform.block(0, 3, 3, 1);
        std::vector<cv::Point> point2DPoseCenter = mathutils::ProjectPoints3DToPoints2D(detectionPoseCenter, camera);
        cv::circle(canvas, point2DPoseCenter[0], nodeSize, cv::Scalar(255), -1, cv::LINE_AA);
    }
}

bool FrameTracker::Do2DTrackingBasedTrack(Frame& currentFrame, const Frame& lastFrame, Mat44_t& velocity) const{
    int nodeSizeHalf = 5;
    int nodeRoiSize = 20;
    int maxTrackSuccessRoiSizeError = 2;
    float maxPoseError = 0.5;

    // Draw nodes for tracking
    cv::Mat nodesCurrentFrame, nodesLastFrame;
    camera::CameraBase myStereoCamera = *_camera;
    DrawNodes(nodesCurrentFrame, currentFrame._threeDDetections, *_camera, nodeSizeHalf);
    DrawNodes(nodesLastFrame, lastFrame._threeDDetections, *_camera, nodeSizeHalf);

    cv::Mat debugTracking = nodesCurrentFrame.clone();
    std::vector<cv::Point3f> objectPoints;
    std::vector<cv::Point2f> imagePoints;
    int countRefObject = 0;
    std::vector<int> indicesCorrespondingDetecton;
    indicesCorrespondingDetecton.reserve(lastFrame._pRefKeyframe->_refObjects.size());
    currentFrame._detectionIDsOfCorrespondingRefObjects.resize(lastFrame._pRefKeyframe->_refObjects.size());
    std::fill(currentFrame._detectionIDsOfCorrespondingRefObjects.begin(), currentFrame._detectionIDsOfCorrespondingRefObjects.end(), -1);
    for (int detectionID : lastFrame._detectionIDsOfCorrespondingRefObjects){
        if (detectionID >= 0){
            ThreeDDetection theDetection = lastFrame._threeDDetections[detectionID];
            Eigen::MatrixXf detectionPoseCenter = Eigen::MatrixXf::Zero(3, 1);
            detectionPoseCenter.block(0, 0, 3, 1) = theDetection._objectInCameraTransform.block(0, 3, 3, 1);
            std::vector<cv::Point> point2DPoseCenter = mathutils::ProjectPoints3DToPoints2D(detectionPoseCenter, myStereoCamera);
            cv::Rect_<double> roiNode(point2DPoseCenter[0].x, point2DPoseCenter[0].y, nodeRoiSize, nodeRoiSize);
            cv::Ptr<cv::legacy::Tracker> twoDTracker = cv::legacy::TrackerMedianFlow::create();
            twoDTracker->init(nodesLastFrame, roiNode);
            cv::Rect_<double> roiNodeUpdate;
            twoDTracker->update(nodesCurrentFrame, roiNodeUpdate);
            TDO_LOG_DEBUG_FORMAT("roiUpdate h, w: %d, %d; nodeRoiSize: %d", roiNodeUpdate.height % roiNodeUpdate.width % nodeRoiSize);
            if (std::abs(roiNodeUpdate.width - nodeRoiSize) < maxTrackSuccessRoiSizeError && std::abs(roiNodeUpdate.height - nodeRoiSize) < maxTrackSuccessRoiSizeError){
                // track success, record all infos.
                cv::Point3f point3D(
                    lastFrame._pRefKeyframe->_refObjects[countRefObject]->_detection._objectInCameraTransform(0, 3),
                    lastFrame._pRefKeyframe->_refObjects[countRefObject]->_detection._objectInCameraTransform(1, 3),
                    lastFrame._pRefKeyframe->_refObjects[countRefObject]->_detection._objectInCameraTransform(2, 3)
                );
                objectPoints.push_back(point3D);
                cv::Point2f point2D(
                    roiNodeUpdate.x,
                    roiNodeUpdate.y
                );
                imagePoints.push_back(point2D);
                // debug image
                cv::rectangle(debugTracking, roiNodeUpdate, cv::Scalar(255), 2, cv::LINE_AA);
                // find inter-frame correspondences
                float bestDistanceToDetectionCenter = std::numeric_limits<float>::infinity();
                int bestIndexDetectionCurrentFrame = -1;
                for (size_t indexDetectionCurrentFrame=0; indexDetectionCurrentFrame < currentFrame._threeDDetections.size(); indexDetectionCurrentFrame++){
                    if (
                        roiNodeUpdate.x > (currentFrame._matchedLeftCamDetections[indexDetectionCurrentFrame]->_centerX - 0.5 * currentFrame._matchedLeftCamDetections[indexDetectionCurrentFrame]->_bWidth)
                        && roiNodeUpdate.x < (currentFrame._matchedLeftCamDetections[indexDetectionCurrentFrame]->_centerX + 0.5 * currentFrame._matchedLeftCamDetections[indexDetectionCurrentFrame]->_bWidth)
                        && roiNodeUpdate.y > (currentFrame._matchedLeftCamDetections[indexDetectionCurrentFrame]->_centerY - 0.5 * currentFrame._matchedLeftCamDetections[indexDetectionCurrentFrame]->_bHeight)
                        && roiNodeUpdate.y < (currentFrame._matchedLeftCamDetections[indexDetectionCurrentFrame]->_centerY + 0.5 * currentFrame._matchedLeftCamDetections[indexDetectionCurrentFrame]->_bHeight)
                    ){
                        Eigen::Vector2f center2DRoiUpdate(roiNodeUpdate.x, roiNodeUpdate.y);
                        Eigen::Vector2f center2DCorrespondingDetectionCurrentFrame(currentFrame._matchedLeftCamDetections[indexDetectionCurrentFrame]->_centerX, currentFrame._matchedLeftCamDetections[indexDetectionCurrentFrame]->_centerY);
                        if (((center2DRoiUpdate - center2DCorrespondingDetectionCurrentFrame).norm() - bestDistanceToDetectionCenter) < bestDistanceToDetectionCenter){
                            bestIndexDetectionCurrentFrame = indexDetectionCurrentFrame;
                            bestDistanceToDetectionCenter = (center2DRoiUpdate - center2DCorrespondingDetectionCurrentFrame).norm();
                        }
                    }
                }
                if (bestIndexDetectionCurrentFrame > 0)
                    currentFrame._detectionIDsOfCorrespondingRefObjects[countRefObject] = bestIndexDetectionCurrentFrame;
            }
        }
        countRefObject++;
    }
    std::filesystem::path debug2DTrackingPath = _sStereoSequencePathForDebug;
    debug2DTrackingPath.append("testBinaryTracking/");
    if (!std::filesystem::exists(debug2DTrackingPath) && !std::filesystem::create_directory(debug2DTrackingPath)){
        TDO_LOG_ERROR_FORMAT("Failed to create the folder: %s", debug2DTrackingPath.string());
        throw std::runtime_error(std::string("Error creating folder: ") + debug2DTrackingPath.string());
    }
    debug2DTrackingPath.append(mathutils::FillZeros(std::to_string(static_cast<int>(currentFrame._timestamp)), 6) + ".png");
    cv::imwrite(debug2DTrackingPath.string(), debugTracking);
    TDO_LOG_DEBUG_FORMAT("written testBinaryTracking debug image: %s", debug2DTrackingPath.string());

    if (objectPoints.size() < 4){
        velocity = Eigen::Matrix4f::Identity();
        currentFrame.SetPose(velocity * lastFrame.GetPose());
        TDO_LOG_DEBUG("2d track fail. not updating camera pose.");
        return false;
    }
    else{
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

        Mat44_t currentFrameInRefKeyFrame;
        TrackWithPnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, maxPoseError, currentFrameInRefKeyFrame);
        TDO_LOG_DEBUG("currentCameraInWorld: \n" << currentFrameInRefKeyFrame);
        if (
            currentFrameInRefKeyFrame.block(0, 3, 3, 1).norm() > maxPoseError
        ){
            // track failed
            velocity = Eigen::Matrix4f::Identity();
            currentFrame.SetPose(velocity * lastFrame.GetPose());
            TDO_LOG_DEBUG("track fail due to too large displacement. not updating camera pose.");
            // not updating cameraInWorld
            return false;
        }
        else{
            velocity = lastFrame.GetPose().inverse() * currentFrameInRefKeyFrame;  //Note: think like there is a point in current frame, first transform it to keyframe, then to last frame.
            currentFrame.SetPose(currentFrameInRefKeyFrame);
            currentFrame._isTracked = true;
            return true;
        }
    }

}


void FrameTracker::CreateNewLandmarks(std::shared_ptr<KeyFrame> pRefKeyFrame, std::shared_ptr<MapDataBase> pMapDb, const std::shared_ptr<object::ObjectBase> pObjectInfo){
    std::vector<std::shared_ptr<LandMark>> visibleLandmarks = pMapDb->GetVisibleLandmarks(pRefKeyFrame);
    float minOverlapAreaRatioForCorrespondence = 0.5;
    std::vector<int> indicesLandmarkForRefObjects(pRefKeyFrame->_refObjects.size(), -1);
    for (int indexRefObject=0; indexRefObject < pRefKeyFrame->_refObjects.size(); indexRefObject++){
        std::shared_ptr<RefObject> pRefObject = pRefKeyFrame->_refObjects[indexRefObject];
        std::vector<cv::Point> refObjectPoints2D = mathutils::ProjectPoints3DToPoints2D(pRefObject->_detection._vertices3DInCamera, (*pRefKeyFrame->_pCamera));
        cv::Mat refObjectPoseMask = mathutils::Draw2DHullMaskFrom2DPointsSet(refObjectPoints2D, (*pRefKeyFrame->_pCamera)._rows, (*pRefKeyFrame->_pCamera)._cols);
        for (int indexVisibleLandmark=0; indexVisibleLandmark < visibleLandmarks.size(); indexVisibleLandmark++){
            std::shared_ptr<LandMark> pOneLandmark = visibleLandmarks[indexVisibleLandmark];
            Eigen::MatrixXf transformedVerticesInWorld = mathutils::TransformPoints<Eigen::MatrixXf>(pOneLandmark->_poseLandmarkInWorld, pOneLandmark->_vertices3DInLandmark);
            Eigen::MatrixXf transformedVerticesInCamera = mathutils::TransformPoints<Eigen::MatrixXf>((pRefKeyFrame->_poseCurrentFrameInWorld).inverse(), transformedVerticesInWorld);
            std::vector<cv::Point> oneLandmarkPoints2D = mathutils::ProjectPoints3DToPoints2D(transformedVerticesInCamera, (*pRefKeyFrame->_pCamera));
            cv::Mat oneLandmarkPoseMask = mathutils::Draw2DHullMaskFrom2DPointsSet(oneLandmarkPoints2D, (*pRefKeyFrame->_pCamera)._rows, (*pRefKeyFrame->_pCamera)._cols);
            cv::Mat overlaps;
            cv::bitwise_and(refObjectPoseMask, oneLandmarkPoseMask, overlaps);
            cv::Scalar overlapArea = cv::sum(overlaps);
            cv::Scalar refObjectPoseMaskArea = cv::sum(refObjectPoseMask);
            if (overlapArea[0] / refObjectPoseMaskArea[0] > minOverlapAreaRatioForCorrespondence){
                indicesLandmarkForRefObjects[indexRefObject] = indexVisibleLandmark;
            }
        }
        if (indicesLandmarkForRefObjects[indexRefObject] < 0){
            // if not correspondence, create landmark.
            Mat44_t poseLandmarkInWorld = pRefKeyFrame->_poseCurrentFrameInWorld * pRefObject->_detection._objectInCameraTransform;
            std::shared_ptr<LandMark> pOneLandmark = std::make_shared<LandMark>(poseLandmarkInWorld, pObjectInfo);
            pOneLandmark->AddObservation(pRefKeyFrame, indexRefObject);
            pMapDb->AddLandMark(pOneLandmark);
        }
        else{
            // if correspondence, and newdetection has higher score, update landmark orientation.
            visibleLandmarks[indicesLandmarkForRefObjects[indexRefObject]]->AddObservation(pRefKeyFrame, indexRefObject);
        }
    }
    TDO_LOG_DEBUG_FORMAT("%d landmarks currently in MapDb.", pMapDb->_landmarks.size());

}

}  // end of namespace eventobjectslam
