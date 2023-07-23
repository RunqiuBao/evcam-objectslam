#include "frame.h"

namespace eventobjectslam {

Frame::Frame(const FrameType frameType, const double timestamp, const std::shared_ptr<camera::CameraBase> pCamera)
: _frameType(frameType), _timestamp(timestamp), _pCamera(pCamera)
{

}

void Frame::SetDetectionsFromExternalSrc(std::vector<TwoDBoundingBox>&& leftCamDetections, std::vector<TwoDBoundingBox>&& rightCamDetections){
    _leftCamDetections = std::move(leftCamDetections);
    _rightCamDetections = std::move(rightCamDetections);
    // // stereo triangulation and template scale based filtering
    _pCamera->MatchStereoBBoxes(_leftCamDetections, _rightCamDetections, _matchedLeftCamDetections, _matchedRightCamDetections);
    // // object 3d bounding box detection and ground plane based pose correction
    _pCamera->CreateThreeDDetections(_matchedLeftCamDetections, _matchedRightCamDetections, (*_leftCamDetections[0]._pObjectInfo), _threeDDetections);
}

std::vector<ThreeDDetection> Frame::Get3DDetections(){
    return _threeDDetections;
}

std::tuple<std::vector<std::shared_ptr<TwoDBoundingBox>>, std::vector<std::shared_ptr<TwoDBoundingBox>>> Frame::GetMatchedDetections(){
    return std::make_tuple(_matchedLeftCamDetections, _matchedRightCamDetections);
}

}  // end of namespace eventobjectslam
