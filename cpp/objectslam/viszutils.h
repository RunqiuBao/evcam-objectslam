#ifndef EVENTOBJECTSLAM_VISZUTILS_H
#define EVENTOBJECTSLAM_VISZUTILS_H

#include <Eigen/Core>
#include <cassert>
#include <opencv2/opencv.hpp>


namespace eventobjectslam {

namespace viszutils {

void Draw3DBoundingBox(
    const Eigen::Ref<const Eigen::Matrix<float, 2, Eigen::Dynamic>, 0, Eigen::Stride<2, 1>> dstPoints,
    cv::Mat& displayImage
){
    assert(displayImage.channels() == 1);
    for (size_t indexVertex=0; indexVertex < dstPoints.cols(); indexVertex++){
        cv::Point vertex = cv::Point(dstPoints(0, indexVertex), dstPoints(1, indexVertex));
        cv::circle(displayImage, vertex, 3, cv::Scalar(0), cv::FILLED);
    }
    std::array<size_t, 12> indicesLineStart = {0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7};
    std::array<size_t, 12> indicesLineEnd = {1, 2, 3, 0, 4, 5, 6, 7, 5, 6, 7, 4};
    for (size_t indexLine=0; indexLine < 12; indexLine++){
        cv::line(displayImage, cv::Point(dstPoints(0, indicesLineStart[indexLine]), dstPoints(1, indicesLineStart[indexLine])), cv::Point(dstPoints(0, indicesLineEnd[indexLine]), dstPoints(1, indicesLineEnd[indexLine])), cv::Scalar(0), 1);
    }

}

/*
 * @param       detections is a (2, 2) shape matrix, first column is 2d coordinates of keypoint1, the second is keypoint2.
 */
void Draw5DDetections(
    const Eigen::Matrix<float, 2, 2>& detections,
    cv::Mat& displayImage 
){
    assert(displayImage.channels() == 3);
    cv::Point keypt1 = cv::Point(detections(0, 0), detections(1, 0));
    cv::circle(displayImage, keypt1, 3, cv::Scalar(0, 255, 0), cv::FILLED);
    cv::Point keypt2 = cv::Point(detections(0, 1), detections(1, 1));
    cv::circle(displayImage, keypt2, 3, cv::Scalar(255, 0, 0), cv::FILLED);
    cv::line(displayImage, keypt1, keypt2, cv::Scalar(0, 255, 255), 2);
}

} // end of viszutils

} // end of eventobjectslam

#endif // EVENTOBJECTSLAM_VISZUTILS_H