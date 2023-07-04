#ifndef EVENTOBJECTSLAM_CAMERA_H
#define EVENTOBJECTSLAM_CAMERA_H

#include <Eigen/Core>

namespace eventobjectslam{

namespace camera{

class CameraBase {

public:
    //! width of image
    const unsigned int _cols;
    //! height of image
    const unsigned int _rows;
    const Eigen::Matrix3f _kk;

    CameraBase(
        const unsigned int cols,
        const unsigned int rows,
        const Eigen::Matrix3f kk
    )
    : _cols(cols), _rows(rows), _kk(kk)
    {}


};

}  // end of namespace camera

}  // end of namespace eventobjectslam

#endif // EVENTOBJECTSLAM_CAMERA_H