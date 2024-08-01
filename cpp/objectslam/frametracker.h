#ifndef EVENTOBJECTSLAM_FRAMETRACKER_H
#define EVENTOBJECTSLAM_FRAMETRACKER_H

#include "objectslam.h"
#include "camera.h"
#include "keyframe.h"
#include "frame.h"
#include "mapdatabase.h"

#include <opencv2/opencv.hpp>

#include <armadillo>

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
        Eigen::VectorXf& outResidual
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
    FrameTracker(std::shared_ptr<camera::CameraBase> camera)
    : _camera(camera), _pPoseOptimizer(std::make_shared<PoseOptimizer>(camera->_kk, camera->_baseline, 3))
    {}

    bool DoMotionBasedTrack(Frame& currentFrame, const Frame& lastFrame, Mat44_t& velocity, const bool isDebug = false) const;

    bool Do2DTrackingBasedTrack(Frame& currentFrame, const Frame& lastFrame, Mat44_t& velocity, const bool isDebug = false) const;

    bool DoRelocalizeFromMap(Frame& currentFrame, const Frame& lastFrame, std::shared_ptr<MapDataBase> pMapDb, Mat44_t& velocity, const bool isDebug);

    void CreateNewLandmarks(std::shared_ptr<KeyFrame> pRefKeyFrame, std::shared_ptr<MapDataBase> mapDb, const bool isDebug = false);

    void SetTrackerStatus(const bool isInitialized){ _isInitialized = isInitialized; }

    bool GetTrackerStatus() { return _isInitialized; }

    std::shared_ptr<KeyFrame> _pRefKeyframe;

    std::string _sStereoSequencePathForDebug;

private:
    std::shared_ptr<camera::CameraBase> _camera;
    std::shared_ptr<PoseOptimizer> _pPoseOptimizer;

    bool _isInitialized = false;


};


};  // end of namespace eventobjectslam

#endif  // EVENTOBJECTSLAM_FRAMETRACKER_H