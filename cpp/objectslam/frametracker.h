#ifndef EVENTOBJECTSLAM_FRAMETRACKER_H
#define EVENTOBJECTSLAM_FRAMETRACKER_H

#include "objectslam.h"
#include "camera.h"
#include "keyframe.h"
#include "frame.h"
#include "mapdatabase.h"

#include <opencv2/opencv.hpp>

#include <armadillo>

struct TrackerParams {
    float minIoUToReject = 0.3f;              // used in tracking methods to build correspondences
    float minIoUToRejectForCloseObject = 0.1f;
    float distanceCloseEnough = 1.5f;
    float maxPoseError = 0.3f;               // used in tracking methods to filter bad track
    float maxPoseErrorInX = 0.3f;
    float maxPoseErrorBA = 0.2f;
    float maxlandmarkErrorBA = 0.6f;
    float maxRotationAngleDeg = 8.0f;         // used in tracking method to filter bad track
};

namespace eventobjectslam{

// tracker state
enum class TrackerStatus {
    NotInitialized,
    Initializing,
    Tracking,
    Lost
};

class PoseOptimizer{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Mat33_t _kk;
    float _baseline;
    int _numLevel;

    PoseOptimizer(const Mat33_t& kk, const float baseline, const int numLevel)
    :_kk(kk), _numLevel(numLevel), _baseline(baseline)
    {}

    void DrawRectangleWithInverseDistance(cv::Mat& featmap, cv::Rect rect, const bool isNormalize=false);

    void PrepareDepthAndFeatureMapFromBboxes(
        std::vector<TwoDBoundingBox>& leftBboxes,
        std::vector<TwoDBoundingBox>& rightBboxes,
        const int imageWidth,
        const int imageHeight,
        cv::Mat& featureMap,
        cv::Mat& depthMap
    );


    void deriveAnalytic(
        const cv::Mat& refDepth,
        const cv::Mat& refImage,
        const std::vector<TwoDBoundingBox>& bboxesRef,
        const cv::Mat& currImage,
        const Vec6_d& xi,
        const Mat33_d kk,
        const int scaleLevel,
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& outJac,
        Eigen::VectorXf& outResidual,
        bool isDebug,
        float opt_rate
    );

    void EstimatePose(
        const cv::Mat& refDepth,
        const cv::Mat& refImage,
        const std::vector<TwoDBoundingBox>& bboxesRef,
        const cv::Mat& currDepth,
        const cv::Mat& currImage,
        Mat44_d& currInRefTransform
    );

};


class FrameTracker {

public:
    // constructor
    FrameTracker(std::shared_ptr<camera::CameraBase> camera, const TrackerParams& params = TrackerParams{})
    : _camera(camera), _params(params), _pPoseOptimizer(std::make_shared<PoseOptimizer>(camera->_kk, camera->_baseline, 3))
    {}

    TrackerParams _params;

    bool DoDenseAlignmentBasedTrack(Frame& currentFrame, const Frame& lastFrame, const bool isDebug) const;

    bool DoMotionBasedTrack(Frame& currentFrame, const Frame& lastFrame, Mat44_t& velocity, const bool isDebug = false) const;

    // Direct versions: correspondences are built by comparing current detection bboxes with past frames' detection
    // bboxes (largest overlap > 0.3 -> same object), walking back frame by frame until the ref keyframe.
    // Pose estimation is done between current detections and the ref objects of the ref keyframe.
    bool DoDenseAlignmentBasedTrackDirect(Frame& currentFrame, const Frame& lastFrame, const bool isDebug) const;

    bool DoMotionBasedTrackDirect(Frame& currentFrame, const Frame& lastFrame, Mat44_t& velocity, const bool isDebug = false) const;

    // Simple 3DoF tracker (x, z, yaw): first estimate the new position from the landmark instances
    // (EstimateTranslationFromLandmarks with the momentum orientation), then at that position estimate the
    // yaw heading: each landmark gives one yaw update w.r.t. the ref keyframe, combined by least squares
    // with equal weight per landmark. Only used when fewer than 3 ref-matched objects are available;
    // otherwise it skips (returns false) so the better-constrained PnP-based tracker runs.
    bool DoSimple3DoFLeastSqauresTrack(Frame& currentFrame, const Frame& lastFrame, Mat44_t& velocity, const bool isDebug = false) const;
    bool DoFacetBasedTrack(Frame& currentFrame, const Frame& lastFrame, Mat44_t& velocity, const bool isDebug, const float minPoseError, const float maxRotationAngleDeg) const;

    bool Do2DTrackingBasedTrack(Frame& currentFrame, const Frame& lastFrame, Mat44_t& velocity, const bool isDebug = false) const;

    bool DoRelocalizeFromMap(Frame& currentFrame, const Frame& lastFrame, std::shared_ptr<MapDataBase> pMapDb, Mat44_t& velocity, const bool isDebug);

    // inheritedLandmarks: per ref-object index, the object instance (landmark) inherited from the tracking
    // association (nullptr if none); inherited instances are kept as-is instead of geometric re-association.
    void CreateNewLandmarks(std::shared_ptr<KeyFrame> pRefKeyFrame, std::shared_ptr<MapDataBase> mapDb, const bool isDebug = false, const std::vector<std::shared_ptr<LandMark>>& inheritedLandmarks = std::vector<std::shared_ptr<LandMark>>());

    void SetTrackerStatus(const bool isInitialized){ _isInitialized = isInitialized; }

    bool GetTrackerStatus() { return _isInitialized; }

    std::shared_ptr<KeyFrame> _pRefKeyframe;

    std::string _sStereoSequencePathForDebug;

private:
    // For each ref object of the ref keyframe, find the index of the corresponding current detection (-1 if none)
    // by chaining 2D bbox overlaps: a current detection is matched to the past detection with the largest bbox IoU
    // (if > 0.3); unresolved detections are checked against earlier frames until the ref keyframe.
    std::vector<int> AssociateDetectionsWithRefObjectsByBboxChain(const Frame& currentFrame) const;

    // Estimate the current frame pose from ref object <-> current detection correspondences (PnP, or landmark
    // translation update when there are too few correspondences). Used by the original motion-based tracking.
    bool TrackPoseWithRefObjectCorrespondences(Frame& currentFrame, const Frame& lastFrame, Mat44_t& velocity, const std::vector<int>& indicesCorrespondingDetecton) const;

    // Collect ref object 3D points and current-frame 2D image points for PnP (object centers; keypoints are
    // added as well when there are fewer than 8 point pairs).
    void CollectPnPCorrespondences(const Frame& currentFrame, const std::vector<int>& indicesCorrespondingDetection, std::vector<cv::Point3f>& objectPoints, std::vector<cv::Point2f>& imagePoints) const;

    // Collect, per matched detection with a live landmark instance, the landmark position in the ref keyframe
    // and the observed object position in the current camera (aligned pairs).
    void CollectLandmarkCorrespondences(const Frame& currentFrame, const std::vector<int>& indicesCorrespondingDetection, std::vector<Vec3_t>& landmarksInRefKeyframe, std::vector<Vec3_t>& landmarksInCurrentCamera) const;

    // Least-squares estimate of the camera translation in the ref keyframe from the landmark instances:
    // with the rotation fixed, each landmark seen at p_cam and mapped at p_kf gives t = p_kf - R * p_cam;
    // the least-squares solution over all instances is the mean. Returns false without any landmark correspondence.
    bool EstimateTranslationFromLandmarks(const Frame& currentFrame, const std::vector<int>& indicesCorrespondingDetection, const Mat33_t& rotationCurrentInRefKeyframe, Vec3_t& translation) const;

    std::shared_ptr<camera::CameraBase> _camera;
    std::shared_ptr<PoseOptimizer> _pPoseOptimizer;

    bool _isInitialized = false;


};


};  // end of namespace eventobjectslam

#endif  // EVENTOBJECTSLAM_FRAMETRACKER_H