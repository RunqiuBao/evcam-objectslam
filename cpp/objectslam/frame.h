#ifndef EVENTOBJECTSLAM_FRAME_H
#define EVENTOBJECTSLAM_FRAME_H

#include "objectslam.h"
#include "object.h"
#include "camera.h"
#include "keyframe.h"

#include <opencv2/opencv.hpp>
#include <numeric>

namespace eventobjectslam{

class KeyFrame;  // Note: due to mutual reference.

enum class FrameType {
    Stereo,
    DepthE  // not supported yet.
};

class Frame {

public:
    Frame(const FrameType frameType, const double timestamp, const std::shared_ptr<camera::CameraBase> pCamera);

    // ~Frame();

    void SetPose(const Mat44_t pose_kc){
        _pose_kc = pose_kc;
    }

    Mat44_t GetPose() const{
        return _pose_kc;
    }

    void SetDetectionsFromExternalSrc(std::vector<TwoDBoundingBox>&& leftCamDetections, std::vector<TwoDBoundingBox>&& rightCamDetections);

    std::vector<ThreeDDetection> Get3DDetections();

    std::tuple<std::vector<std::shared_ptr<TwoDBoundingBox>>, std::vector<std::shared_ptr<TwoDBoundingBox>>> GetMatchedDetections();

    void Refine3DDetections();

    void Draw3DVerticesFor3DDetections(
        const std::shared_ptr<object::ObjectBase> pObjectInfo,
        const std::shared_ptr<eventobjectslam::camera::CameraBase> pCamera,
        std::vector<ThreeDDetection>& threeDDetections
    );

    void SetDetectionsAsRefObjects(){
        _detectionIDsOfCorrespondingRefObjects.resize(_threeDDetections.size());
        std::iota(_detectionIDsOfCorrespondingRefObjects.begin(), _detectionIDsOfCorrespondingRefObjects.end(), 0);
    }

    std::shared_ptr<KeyFrame> _pRefKeyframe;
    // 3D detections from the matched 2D detections
    std::vector<ThreeDDetection> _threeDDetections;
    std::vector<int> _detectionIDsOfCorrespondingRefObjects; // Note: size if same as refObjects in refKeyframe; if not correspondence, will be -1.
    // 2D detections on the left- and right Image
    std::vector<TwoDBoundingBox> _leftCamDetections;
    std::vector<TwoDBoundingBox> _rightCamDetections;
    // matched 2D detections
    std::vector<std::shared_ptr<TwoDBoundingBox>> _matchedLeftCamDetections;  // Note: same order as _threeDDetections
    std::vector<std::shared_ptr<TwoDBoundingBox>> _matchedRightCamDetections;
    double _timestamp;

    bool _isTracked = false; // Note: whether tracking succeeded or not on this frame.

private:
    FrameType _frameType;
    cv::Mat _leftImage;
    cv::Mat _rightImage;
    cv::Mat _maskImage;
    // frame pose
    Mat44_t _pose_kc;  // current in refKeyrame transform
    // camera instance that took this frame
    std::shared_ptr<camera::CameraBase> _pCamera;

};


}  // end of namespace eventobjectslam

#endif  // EVENTOBJECTSLAM_FRAME_H