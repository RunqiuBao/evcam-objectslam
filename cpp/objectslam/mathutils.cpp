#include "mathutils.h"

namespace eventobjectslam {

std::vector<cv::Point> mathutils::ProjectPoints3DToPoints2D(Eigen::MatrixXf& mPoints3D, camera::CameraBase& camera){
    Eigen::Matrix<float, 2, Eigen::Dynamic> mPoints2D = Eigen::Matrix<float, 2, Eigen::Dynamic>::Zero(2, mPoints3D.cols());
    // TDO_LOG_DEBUG_FORMAT("mPoints3D rows: %d, mPoints3D cols: %d", mPoints3D.rows() % mPoints3D.cols());
    camera.ProjectPoints(
        mPoints3D,
        mPoints2D
    );
    std::vector<cv::Point> points2DCV;
    for (size_t indexPoint = 0; indexPoint < mPoints2D.cols(); indexPoint++){
        points2DCV.push_back(cv::Point(static_cast<int32_t>(mPoints2D(0, indexPoint)), static_cast<int32_t>(mPoints2D(1, indexPoint))));
    }
    return points2DCV;
}

cv::Mat mathutils::Draw2DHullMaskFrom2DPointsSet(const std::vector<cv::Point>& points2DCV, const size_t imageH, const size_t imageW){
    cv::Mat hullMask = cv::Mat::zeros(imageH, imageW, CV_8UC1);
    std::vector<cv::Point> hullPoints2DCV;
    cv::convexHull(points2DCV, hullPoints2DCV);
    std::vector<std::vector<cv::Point>> vHullPoints2DCV;
    vHullPoints2DCV.push_back(hullPoints2DCV);
    cv::drawContours(hullMask, vHullPoints2DCV, -1, cv::Scalar(1), cv::FILLED);
    return hullMask;
}

}  // end of eventobjectslam