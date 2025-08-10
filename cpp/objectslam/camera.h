#ifndef EVENTOBJECTSLAM_CAMERA_H
#define EVENTOBJECTSLAM_CAMERA_H

#include <Eigen/Core>
#include <Eigen/Dense>  // Note: need to use matrix inverse
#include "objectslam.h"
#include "object.h"

namespace eventobjectslam{

namespace camera{

const float maxVisiableDistance = std::numeric_limits<float>::max(); // no limit for now.

class CameraBase {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    unsigned int _cameraID;
    //! width of image
    unsigned int _cols;
    //! height of image
    unsigned int _rows;
    Eigen::Matrix3f _kk;
    std::vector<float> _distortCoef;
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
        const std::vector<TwoDBoundingBox>& leftCamDetections,
        const std::vector<TwoDBoundingBox>& rightCamDetections,
        std::vector<std::shared_ptr<TwoDBoundingBox>>& matchedLeftCamDetections,
        std::vector<std::shared_ptr<TwoDBoundingBox>>& matchedRightCamDetections
    );

    float TriangulateDepthInStereoCamera(const float disparity);

    void ProjectPointTo3DByDepth(const float Z, const float u, const float v, float& X, float& Y);

    void ProjectPoints(
        const Eigen::Ref<const Eigen::Matrix<float, 3, Eigen::Dynamic>> points,
        Eigen::Ref<Eigen::Matrix<float, 2, Eigen::Dynamic>> dstPoints
    );

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
