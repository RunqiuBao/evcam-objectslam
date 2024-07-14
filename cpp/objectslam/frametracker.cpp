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

bool FrameTracker::DoMotionBasedTrack(Frame& currentFrame, const Frame& lastFrame, Mat44_t& velocity, const bool isDebug) const{
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
        Eigen::MatrixXf transformedVertices = mathutils::TransformPoints<Eigen::MatrixXf>((velocity * lastFrame.GetPose()).inverse(), refObject->_detection._vertices3DInRefFrame);
        std::vector<cv::Point> points2DCV = mathutils::ProjectPoints3DToPoints2D(transformedVertices, myStereoCamera);
        cv::Mat refObjectPoseMask = mathutils::Draw2DHullMaskFrom2DPointsSet(points2DCV, myStereoCamera._rows, myStereoCamera._cols);
        int refObjectPoseMaskDataType = refObjectPoseMask.type();
        int displayRefObjectsDataType = displayRefObjects.type();
        cv::bitwise_or(refObjectPoseMask, displayRefObjects, displayRefObjects);
        size_t countDetection = 0;
        int indexLargestOverlap = -1;
        int largestOverlapArea = 0;
        for (ThreeDDetection currentDetection : currentFrame._threeDDetections){
            std::vector<cv::Point> points2DCV = mathutils::ProjectPoints3DToPoints2D(currentDetection._vertices3DInRefFrame, myStereoCamera);
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
            std::vector<cv::Point> points2DCV = mathutils::ProjectPoints3DToPoints2D(currentFrame._threeDDetections[indexLargestOverlap]._vertices3DInRefFrame, myStereoCamera);
            cv::Mat detectionPoseMask = mathutils::Draw2DHullMaskFrom2DPointsSet(points2DCV, myStereoCamera._rows, myStereoCamera._cols);
            cv::bitwise_or(detectionPoseMask, displayDetections, displayDetections);
            TDO_LOG_DEBUG("found corresponding detection for refObject " << std::to_string(countRefObject) << ", overlap area :" << largestOverlapArea);
        }
        indicesCorrespondingDetecton.push_back(indexLargestOverlap);

        countRefObject++;
    }

    if (isDebug){
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
    }

    // 3D object points in world coordinates
    std::vector<cv::Point3f> objectPoints;
    // Populate objectPoints with the corresponding 3D points from the object
    // 2D image points in image coordinates
    std::vector<cv::Point2f> imagePoints;
    // Populate imagePoints with the corresponding 2D points from the object in the image
    size_t indexRefObject = 0;
    for (int indexCorrespondingDetection : indicesCorrespondingDetecton){
        if (indexCorrespondingDetection < 0){
            indexRefObject++;
            continue;
        }
        // object center
        cv::Point3f point3D(
            refObjects[indexRefObject]->_detection._objectCenterInRefFrame(0),
            refObjects[indexRefObject]->_detection._objectCenterInRefFrame(1),
            refObjects[indexRefObject]->_detection._objectCenterInRefFrame(2)
        );
        objectPoints.push_back(point3D);
        cv::Point2f point2D(
            (*currentFrame._matchedLeftCamDetections[indexCorrespondingDetection])._centerX,
            (*currentFrame._matchedLeftCamDetections[indexCorrespondingDetection])._centerY
        );
        imagePoints.push_back(point2D);
        // // keypt
        // cv::Point3f point3D_keypt(
        //     refObjects[indexRefObject]->_detection._keypt1InRefFrame(0),
        //     refObjects[indexRefObject]->_detection._keypt1InRefFrame(1),
        //     refObjects[indexRefObject]->_detection._keypt1InRefFrame(2)
        // );
        // objectPoints.push_back(point3D_keypt);
        // cv::Point2f point2D_keypt(
        //     (*currentFrame._matchedLeftCamDetections[indexCorrespondingDetection])._keypts[0][0],
        //     (*currentFrame._matchedLeftCamDetections[indexCorrespondingDetection])._keypts[0][1]
        // );
        // imagePoints.push_back(point2D_keypt);
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
        Eigen::MatrixXf detectionPoints3D = Eigen::MatrixXf::Zero(3, 1);
        detectionPoints3D.col(0) = oneDetection._objectCenterInRefFrame;
        std::vector<cv::Point> point2DPoseCenter = mathutils::ProjectPoints3DToPoints2D(detectionPoints3D, camera);

        cv::circle(canvas, point2DPoseCenter[0], nodeSize, cv::Scalar(255), -1, cv::LINE_AA);
    }
}

bool FrameTracker::Do2DTrackingBasedTrack(Frame& currentFrame, const Frame& lastFrame, Mat44_t& velocity, const bool isDebug) const{
    int nodeSizeHalf = 5;
    int nodeRoiSize = 20;
    int maxTrackSuccessRoiSizeError = 2;
    float maxPoseError = 0.5;

    // Draw nodes for tracking
    cv::Mat nodesCurrentFrame, nodesLastFrame;
    camera::CameraBase myStereoCamera = *_camera;
    DrawNodes(nodesCurrentFrame, currentFrame._threeDDetections, *_camera, nodeSizeHalf);
    DrawNodes(nodesLastFrame, lastFrame._threeDDetections, *_camera, nodeSizeHalf);

    cv::Mat debugTracking;
    if (isDebug){
        debugTracking = nodesCurrentFrame.clone();
    }
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
            detectionPoseCenter.col(0) = theDetection._objectCenterInRefFrame;
            std::vector<cv::Point> point2DPoseCenter = mathutils::ProjectPoints3DToPoints2D(detectionPoseCenter, myStereoCamera);
            cv::Rect_<double> roiNode(point2DPoseCenter[0].x, point2DPoseCenter[0].y, nodeRoiSize, nodeRoiSize);
            cv::Ptr<cv::legacy::Tracker> twoDTracker = cv::legacy::TrackerMedianFlow::create();
            twoDTracker->init(nodesLastFrame, roiNode);
            cv::Rect_<double> roiNodeUpdate;
            twoDTracker->update(nodesCurrentFrame, roiNodeUpdate);
            TDO_LOG_DEBUG_FORMAT("roiUpdate h, w: %d, %d; nodeRoiSize: %d", roiNodeUpdate.height % roiNodeUpdate.width % nodeRoiSize);
            if (std::abs(roiNodeUpdate.width - nodeRoiSize) < maxTrackSuccessRoiSizeError && std::abs(roiNodeUpdate.height - nodeRoiSize) < maxTrackSuccessRoiSizeError){
                // track success, record all infos.
                // // object center
                cv::Point3f point3D(
                    lastFrame._pRefKeyframe->_refObjects[countRefObject]->_detection._objectCenterInRefFrame(0),
                    lastFrame._pRefKeyframe->_refObjects[countRefObject]->_detection._objectCenterInRefFrame(1),
                    lastFrame._pRefKeyframe->_refObjects[countRefObject]->_detection._objectCenterInRefFrame(2)
                );
                objectPoints.push_back(point3D);
                cv::Point2f point2D(
                    roiNodeUpdate.x,
                    roiNodeUpdate.y
                );
                imagePoints.push_back(point2D);

                if (isDebug){
                    // debug image
                    cv::rectangle(debugTracking, roiNodeUpdate, cv::Scalar(255), 2, cv::LINE_AA);
                }
                // find intercv::Mat debugTracking-frame correspondences
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

    if(isDebug){
        std::filesystem::path debug2DTrackingPath = _sStereoSequencePathForDebug;
        debug2DTrackingPath.append("testBinaryTracking/");
        if (!std::filesystem::exists(debug2DTrackingPath) && !std::filesystem::create_directory(debug2DTrackingPath)){
            TDO_LOG_ERROR_FORMAT("Failed to create the folder: %s", debug2DTrackingPath.string());
            throw std::runtime_error(std::string("Error creating folder: ") + debug2DTrackingPath.string());
        }
        debug2DTrackingPath.append(mathutils::FillZeros(std::to_string(static_cast<int>(currentFrame._timestamp)), 6) + ".png");
        cv::imwrite(debug2DTrackingPath.string(), debugTracking);
        TDO_LOG_DEBUG_FORMAT("written testBinaryTracking debug image: %s", debug2DTrackingPath.string());
    }

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


void FrameTracker::CreateNewLandmarks(std::shared_ptr<KeyFrame> pRefKeyFrame, std::shared_ptr<MapDataBase> pMapDb){
    std::vector<std::shared_ptr<LandMark>> visibleLandmarks = pMapDb->GetVisibleLandmarks(pRefKeyFrame);  // Note: find landmarks that might fall within FoV of this keyframe.
    std::map<std::shared_ptr<LandMark>, unsigned int> observedLandmarks_indicesRefObj;
    float minOverlapAreaRatioForCorrespondence = 0.5;
    std::vector<int> indicesLandmarkForRefObjects(pRefKeyFrame->_refObjects.size(), -1);
    std::vector<float> distancesObjectToClosestLandmark(pRefKeyFrame->_refObjects.size(), std::numeric_limits<float>::max());
    std::vector<int> indicesForClosestLandmark(pRefKeyFrame->_refObjects.size(), -1);
    size_t countNewLandmark = 0;
    for (int indexRefObject=0; indexRefObject < pRefKeyFrame->_refObjects.size(); indexRefObject++){
        std::shared_ptr<RefObject> pRefObject = pRefKeyFrame->_refObjects[indexRefObject];
        std::vector<cv::Point> refObjectPoints2D = mathutils::ProjectPoints3DToPoints2D(pRefObject->_detection._vertices3DInRefFrame, (*pRefKeyFrame->_pCamera));
        cv::Mat refObjectPoseMask = mathutils::Draw2DHullMaskFrom2DPointsSet(refObjectPoints2D, (*pRefKeyFrame->_pCamera)._rows, (*pRefKeyFrame->_pCamera)._cols);
        for (int indexVisibleLandmark=0; indexVisibleLandmark < visibleLandmarks.size(); indexVisibleLandmark++){
            std::shared_ptr<LandMark> pOneLandmark = visibleLandmarks[indexVisibleLandmark];
            Eigen::MatrixXf transformedVerticesInWorld = mathutils::TransformPoints<Eigen::MatrixXf>(pOneLandmark->GetLandmarkPoseInWorld(), pOneLandmark->_vertices3DInLandmark);
            Eigen::MatrixXf transformedVerticesInCamera = mathutils::TransformPoints<Eigen::MatrixXf>((pRefKeyFrame->GetKeyframePoseInWorld()).inverse(), transformedVerticesInWorld);
            std::vector<cv::Point> oneLandmarkPoints2D = mathutils::ProjectPoints3DToPoints2D(transformedVerticesInCamera, (*pRefKeyFrame->_pCamera));
            cv::Mat oneLandmarkPoseMask = mathutils::Draw2DHullMaskFrom2DPointsSet(oneLandmarkPoints2D, (*pRefKeyFrame->_pCamera)._rows, (*pRefKeyFrame->_pCamera)._cols);
            cv::Mat overlaps;
            cv::bitwise_and(refObjectPoseMask, oneLandmarkPoseMask, overlaps);
            cv::Scalar overlapArea = cv::sum(overlaps);
            cv::Scalar refObjectPoseMaskArea = cv::sum(refObjectPoseMask);
            if ((overlapArea[0] / refObjectPoseMaskArea[0]) > minOverlapAreaRatioForCorrespondence){
                indicesLandmarkForRefObjects[indexRefObject] = indexVisibleLandmark;
                observedLandmarks_indicesRefObj[visibleLandmarks[indexVisibleLandmark]] = indexRefObject;
                break;
            }
            else if (overlapArea[0] > 0) {
                // TODO: need collision check here.
                Vec3_t objectCenterInWorld = pRefKeyFrame->GetKeyframePoseInWorld().block<3, 3>(0, 0) * pRefObject->_detection._objectCenterInRefFrame + pRefKeyFrame->GetKeyframePoseInWorld().col(3).head<3>();
                Mat44_t poseExistingLandmark = visibleLandmarks[indexVisibleLandmark]->GetLandmarkPoseInWorld();
                Eigen::Vector3f vObjectToLandmark = objectCenterInWorld - poseExistingLandmark.col(3).head<3>();
                float distanceO2L = vObjectToLandmark.norm();
                distancesObjectToClosestLandmark[indexRefObject] = distanceO2L;
                indicesForClosestLandmark[indexRefObject] = indexVisibleLandmark;
            }
            else{
                continue;
            }
        }
        float distanceThreshold;
        if (indicesForClosestLandmark[indexRefObject] >= 0){
            // found a closest landmark.
            distanceThreshold = visibleLandmarks[indicesForClosestLandmark[indexRefObject]]->_horizontalSize * 3.;  // Note: 3.0 is a factor.
        }
        else{
            // might be the initial keyframe. Or there are no visiable landmarks existing for this keyframe.
            distanceThreshold =  0;
        }
        if (indicesLandmarkForRefObjects[indexRefObject] < 0){
            if (distancesObjectToClosestLandmark[indexRefObject] > distanceThreshold){  // if within certain physical distance, still create correspondence.
                // if not correspondence, create landmark.
                Mat44_t poseLandmarkInWorld;
                Vec3_t keypt1InLandmark;
                LandMark::ComputeLandmarkPoseInWorldAndKeypt1InWolrd(pRefKeyFrame, pRefObject, poseLandmarkInWorld, keypt1InLandmark);
                std::shared_ptr<object::ObjectBase> pObjectInfo = pRefObject->_detection._pObjectInfo;
                std::shared_ptr<LandMark> pOneLandmark = std::make_shared<LandMark>(poseLandmarkInWorld, keypt1InLandmark, pRefObject->_detection._horizontalSize, pObjectInfo);
                pOneLandmark->AddObservation(pRefKeyFrame, indexRefObject);
                pMapDb->AddLandMark(pOneLandmark);
                observedLandmarks_indicesRefObj[pOneLandmark] = indexRefObject;
                TDO_LOG_DEBUG_FORMAT("Failed matching correspondence (distance %f). Creating new landmark...", distancesObjectToClosestLandmark[indexRefObject]);
                countNewLandmark++;
                pRefKeyFrame->_bContainNewLandmarks = true;
                continue;
            }
            indicesLandmarkForRefObjects[indexRefObject] = indicesForClosestLandmark[indexRefObject];
            TDO_LOG_DEBUG_FORMAT("Resurrect correspondence due to close 3d distance (%f m).", distancesObjectToClosestLandmark[indexRefObject]);
        }
        // if correspondence, and new keyframe is closer, update landmark pose.
        // TODO: should use multi-view stereo to update landmark pose. need stereo rectification and check if object is within FoV after stereo recti.
        visibleLandmarks[indicesLandmarkForRefObjects[indexRefObject]]->AddObservation(pRefKeyFrame, indexRefObject);
        observedLandmarks_indicesRefObj[visibleLandmarks[indicesLandmarkForRefObjects[indexRefObject]]] = indexRefObject;
    }
    pRefKeyFrame->InitializeObservedLandmarks(observedLandmarks_indicesRefObj);
    TDO_LOG_INFO_FORMAT("Created %d new landmarks in keyframe %d. \nTotally %d landmarks currently in MapDb.", countNewLandmark % pRefKeyFrame->_keyFrameID % pMapDb->_landmarks.size());

}

}  // end of namespace eventobjectslam
