#include "frame.h"

#include <numeric>

#include <logging.h>
TDO_LOGGER("eventobjectslam.frame")

namespace eventobjectslam {

Frame::Frame(const FrameType frameType, const double timestamp, const std::shared_ptr<camera::CameraBase> pCamera)
: _frameType(frameType), _timestamp(timestamp), _pCamera(pCamera)
{

}

static ThreeDPlane FitPlaneBySVD(std::vector<Eigen::Vector3f> pointsOnPlaneInliers){
    Eigen::MatrixXf mPointsOnPlane(3, pointsOnPlaneInliers.size());
    for (size_t indexPoint = 0; indexPoint < pointsOnPlaneInliers.size(); indexPoint++){
        mPointsOnPlane.col(indexPoint) = pointsOnPlaneInliers[indexPoint];
    }
    Eigen::Vector3f centerPoint = mPointsOnPlane.rowwise().mean();
    Eigen::MatrixXf mCenteredPointsOnPlane(3, pointsOnPlaneInliers.size());
    for (size_t indexPoint = 0; indexPoint < pointsOnPlaneInliers.size(); indexPoint++){
        mCenteredPointsOnPlane.col(indexPoint) = mPointsOnPlane.block(0, indexPoint, 3, 1) - centerPoint;
    }
    Eigen::JacobiSVD<Eigen::MatrixXf> svdOperator(mCenteredPointsOnPlane, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXf singularValues = svdOperator.singularValues();
    Eigen::MatrixXf leftSingularVectors = svdOperator.matrixU();
    Eigen::MatrixXf rightSingularVectors = svdOperator.matrixV();
    TDO_LOG_DEBUG_FORMAT("leftSingularVectors shape: h * w: %f * %f", leftSingularVectors.rows() % leftSingularVectors.cols());
    Eigen::Vector3f normalPlane;
    normalPlane << leftSingularVectors(0, 2), leftSingularVectors(1, 2), leftSingularVectors(2, 2);
    if (normalPlane[1] > 0)
        normalPlane *= -1;  // Y direction need to be negative.
    ThreeDPlane planeModel = {centerPoint[0], centerPoint[1], centerPoint[2], normalPlane[0], normalPlane[1], normalPlane[2]};
    return planeModel;

} 

static float ComputeAvgDistanceOfPointsToPlane(ThreeDPlane planeModel, std::vector<Eigen::Vector3f> pointsOnPlaneInliers){
    float averageDistance = 0.;
    Eigen::Vector3f centerPoint;
    centerPoint << planeModel[0], planeModel[1], planeModel[2];
    Eigen::Vector3f normalPlane;
    normalPlane << planeModel[3], planeModel[4], planeModel[5];
    for (Eigen::Vector3f point : pointsOnPlaneInliers){
        averageDistance += std::abs((point - centerPoint).dot(normalPlane));
    }
    return averageDistance / pointsOnPlaneInliers.size();
}

void Frame::Refine3DDetections(){
    if (_threeDDetections.size() <= 3){
        return;
    }
    // (1) refine by ground plane.
    int numSamplePoints = 3;
    int maxNumIterations = 6;
    float inlierDistanceToPlaneThreshold = 0.1;
    int minNumPointsInliers = 1;
    ThreeDPlane planeModelBestFit;
    bool isSuccess = false;
    int numFinalInliers = 0;
    float bestAvgDistanceToPlane = std::numeric_limits<float>::infinity();
    float averageDistanceToPlane = 0.;
    for(size_t indexIteration=0; indexIteration < maxNumIterations; indexIteration++){
        std::vector<int> pointIndicies(_threeDDetections.size());
        std::iota(pointIndicies.begin(), pointIndicies.end(), 0);
        cv::randShuffle(pointIndicies);
        std::vector<std::shared_ptr<ThreeDDetection>> detectionsOnPlaneInliers;
        std::vector<Eigen::Vector3f> pointsOnPlaneInliers;
        detectionsOnPlaneInliers.reserve(numSamplePoints);
        pointsOnPlaneInliers.reserve(numSamplePoints);
        for (int count=0; count < numSamplePoints; count++){
            detectionsOnPlaneInliers.push_back(std::make_shared<ThreeDDetection>(_threeDDetections[pointIndicies[count]]));
            pointsOnPlaneInliers.push_back(_threeDDetections[pointIndicies[count]]._objectInCameraTransform.block(0, 3, 3, 1));
        }
        Eigen::MatrixXf mPointsOnPlane(3, pointsOnPlaneInliers.size());
        for (size_t indexPoint = 0; indexPoint < pointsOnPlaneInliers.size(); indexPoint++){
            mPointsOnPlane.col(indexPoint) = pointsOnPlaneInliers[indexPoint];
        }
        Eigen::Vector3f centerPoint = mPointsOnPlane.rowwise().mean();
        Eigen::Vector3f normalPlane = (pointsOnPlaneInliers[1] - pointsOnPlaneInliers[0]).cross(pointsOnPlaneInliers[2] - pointsOnPlaneInliers[0]);
        normalPlane /= normalPlane.norm();
        if (normalPlane[1] > 0)
            normalPlane *= -1;  // Y direction need to be negative.
        ThreeDPlane planeModel = {centerPoint[0], centerPoint[1], centerPoint[2], normalPlane[0], normalPlane[1], normalPlane[2]};
        std::vector<Eigen::Vector3f> pointsOnPlaneInliersAdd;
        for(int count=numSamplePoints; count < _threeDDetections.size(); count++){
            Eigen::Vector3f pointCandidate = _threeDDetections[count]._objectInCameraTransform.block(0, 3, 3, 1);
            float distanceError = std::abs((pointCandidate - centerPoint).dot(normalPlane));
            TDO_LOG_DEBUG_FORMAT("iteration %d, distanceError %f m", indexIteration % distanceError);
            if(distanceError < inlierDistanceToPlaneThreshold){
                pointsOnPlaneInliersAdd.push_back(pointCandidate);
            }
        }
        if (pointsOnPlaneInliersAdd.size() >= minNumPointsInliers){
            for (Eigen::Vector3f pointCandidate : pointsOnPlaneInliersAdd){
                pointsOnPlaneInliers.push_back(pointCandidate);
            }
            // plane fitting by svd
            planeModel = FitPlaneBySVD(pointsOnPlaneInliers);
            centerPoint << planeModel[0], planeModel[1], planeModel[2];
            normalPlane << planeModel[3], planeModel[4], planeModel[5];
            averageDistanceToPlane = ComputeAvgDistanceOfPointsToPlane(planeModel, pointsOnPlaneInliers);
            if (averageDistanceToPlane < bestAvgDistanceToPlane){
                bestAvgDistanceToPlane = averageDistanceToPlane;
                planeModelBestFit = planeModel;
                isSuccess = true;
                numFinalInliers = pointsOnPlaneInliers.size();
            }
        }
    }
    if (isSuccess){
        TDO_LOG_DEBUG("plane fitting succeeded. averageDistanceToPlane: " << averageDistanceToPlane << ", num inliers: " << numFinalInliers);
        TDO_LOG_DEBUG("planeModelBestFit: " << planeModelBestFit[0] << ", " << planeModelBestFit[1] << ", " << planeModelBestFit[2] << ", " << planeModelBestFit[3] << ", " << planeModelBestFit[4] << ", " << planeModelBestFit[5]);
    }
    else{
        TDO_LOG_DEBUG("plane fitting failed. averageDistanceToPlane: " << averageDistanceToPlane);
    }

}

void Frame::SetDetectionsFromExternalSrc(std::vector<TwoDBoundingBox>&& leftCamDetections, std::vector<TwoDBoundingBox>&& rightCamDetections){
    _leftCamDetections = std::move(leftCamDetections);
    _rightCamDetections = std::move(rightCamDetections);
    // // stereo triangulation and template scale based filtering
    _pCamera->MatchStereoBBoxes(_leftCamDetections, _rightCamDetections, _matchedLeftCamDetections, _matchedRightCamDetections);
    // // object 3d bounding box detection and ground plane based pose correction
    _pCamera->CreateThreeDDetections(_matchedLeftCamDetections, _matchedRightCamDetections, (*_leftCamDetections[0]._pObjectInfo), _threeDDetections);

    Refine3DDetections();
    
    
}

std::vector<ThreeDDetection> Frame::Get3DDetections(){
    return _threeDDetections;
}

std::tuple<std::vector<std::shared_ptr<TwoDBoundingBox>>, std::vector<std::shared_ptr<TwoDBoundingBox>>> Frame::GetMatchedDetections(){
    return std::make_tuple(_matchedLeftCamDetections, _matchedRightCamDetections);
}

}  // end of namespace eventobjectslam
