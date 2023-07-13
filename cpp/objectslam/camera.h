#ifndef EVENTOBJECTSLAM_CAMERA_H
#define EVENTOBJECTSLAM_CAMERA_H

#include <Eigen/Core>

namespace eventobjectslam{

namespace camera{

class CameraBase {

public:
    //! width of image
    unsigned int _cols;
    //! height of image
    unsigned int _rows;
    Eigen::Matrix3f _kk;
    float _baseline;

    CameraBase(
        const unsigned int cols,
        const unsigned int rows,
        const Eigen::Matrix3f kk,
        const float baseline
    )
    : _cols(cols), _rows(rows), _kk(kk), _baseline(baseline)
    {}

};

}  // end of namespace camera

}  // end of namespace eventobjectslam

#endif // EVENTOBJECTSLAM_CAMERA_H