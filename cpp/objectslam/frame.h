#ifndef EVENTOBJECTSLAM_FRAME_H
#define EVENTOBJECTSLAM_FRAME_H

#include "objectslam.h"
#include "object.h"
#include "camera.h"

#include <opencv2/opencv.hpp>

namespace eventobjectslam{

enum class FrameType {
    Stereo,
    DepthE  // not supported yet.
};

class Frame {

public:
    Frame(const FrameType frameType, const double timestamp, const std::shared_ptr<camera::CameraBase> pCamera);

    // ~Frame();
    void SetDetectionsFromExternalSrc(std::vector<TwoDBoundingBox>&& leftCamDetections, std::vector<TwoDBoundingBox>&& rightCamDetections);

    std::vector<ThreeDDetection> Get3DDetections();

    std::tuple<std::vector<std::shared_ptr<TwoDBoundingBox>>, std::vector<std::shared_ptr<TwoDBoundingBox>>> GetMatchedDetections();

    void Refine3DDetections();

    void Draw3DVerticesFor3DDetections(
        const object::ObjectBase& objectInfo,
        const std::shared_ptr<eventobjectslam::camera::CameraBase> pCamera,
        std::vector<ThreeDDetection>& threeDDetections
    );

private:
    FrameType _frameType;
    cv::Mat _leftImage;
    cv::Mat _rightImage;
    cv::Mat _maskImage;
    double _timestamp;
    // 2D detections on the left- and right Image
    std::vector<TwoDBoundingBox> _leftCamDetections;
    std::vector<TwoDBoundingBox> _rightCamDetections;
    // matched 2D detections
    std::vector<std::shared_ptr<TwoDBoundingBox>> _matchedLeftCamDetections;
    std::vector<std::shared_ptr<TwoDBoundingBox>> _matchedRightCamDetections;
    // frame pose
    Mat44_t _pose;
    // camera instance that took this frame
    std::shared_ptr<camera::CameraBase> _pCamera;
    // 3D detections from the matched 2D detections
    std::vector<ThreeDDetection> _threeDDetections;

};


}  // end of namespace eventobjectslam

#endif  // EVENTOBJECTSLAM_FRAME_H