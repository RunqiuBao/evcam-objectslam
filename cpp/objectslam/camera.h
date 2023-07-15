#ifndef EVENTOBJECTSLAM_CAMERA_H
#define EVENTOBJECTSLAM_CAMERA_H

#include <Eigen/Core>
#include <Eigen/Dense>  // Note: need to use matrix inverse
#include "objectslam.h"
#include "object.h"

namespace eventobjectslam{

namespace camera{

class CameraBase {

public:
    unsigned int _cameraID;
    //! width of image
    unsigned int _cols;
    //! height of image
    unsigned int _rows;
    Eigen::Matrix3f _kk;
    float _baseline;

    CameraBase(
        const unsigned int cameraID,
        const unsigned int cols,
        const unsigned int rows,
        const Eigen::Matrix3f kk,
        const float baseline
    )
    : _cameraID(cameraID), _cols(cols), _rows(rows), _kk(kk), _baseline(baseline)
    {}

    void MatchStereoBBoxes(
        std::vector<TwoDBoundingBox>& leftCamDetections,
        std::vector<TwoDBoundingBox>& rightCamDetections,
        const object::ObjectBase& objectInfo,
        std::vector<std::shared_ptr<TwoDBoundingBox>>& matchedLeftCamDetections,
        std::vector<std::shared_ptr<TwoDBoundingBox>>& matchedRightCamDetections
    );

    float TriangulateThreeDPointInStereoCamera(const float disparity);

    Eigen::MatrixXf ProjectPoints(Eigen::MatrixXf points);

    void CreateThreeDDetections(
        const std::vector<std::shared_ptr<TwoDBoundingBox>>& matchedLeftCamDetections,
        const std::vector<std::shared_ptr<TwoDBoundingBox>>& matchedRightCamDetections,
        const object::ObjectBase& objectInfo,
        std::vector<ThreeDDetection>& threeDDetections
    );

};

}  // end of namespace camera

}  // end of namespace eventobjectslam

#endif // EVENTOBJECTSLAM_CAMERA_H