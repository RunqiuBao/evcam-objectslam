#include "mathutils.h"

#include <random>

#include <logging.h>
TDO_LOGGER("objectslam.mathutils")

namespace eventobjectslam {

void mathutils::FilterNonPlanePoints(
    const std::vector<Vec3_t> points,
    const float planeDistanceThreshold,
    std::vector<int>& indicesPoints
){
    if (points.size() < 5) {
        indicesPoints.resize(points.size());
        std::iota(indicesPoints.begin(), indicesPoints.end(), 0);
        return;
    }
    pcl::PointCloud<pcl::PointXYZ>::Ptr pPoints(new pcl::PointCloud<pcl::PointXYZ>);
    pPoints->width = 1;
    pPoints->height = points.size();
    pPoints->points.resize(points.size());
    for (size_t indexPoint = 0; indexPoint < points.size(); indexPoint++){
        pPoints->points[indexPoint] = pcl::PointXYZ(points[indexPoint](0), points[indexPoint](1), points[indexPoint](2));
    }
    // use pcl ransac segmentation to estimate the plane
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(3);
    seg.setDistanceThreshold(planeDistanceThreshold);  // unit is meter.
    pcl::ModelCoefficients::Ptr planeCoeff(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr pIndicesInliers(new pcl::PointIndices);
    
    seg.setInputCloud(pPoints);
    seg.segment(*pIndicesInliers, *planeCoeff);

    if (pIndicesInliers->indices.size() == 0){
        TDO_LOG_DEBUG_FORMAT("plane fitting zero inliers from %d points, plane distance threshold = %f", points.size() % planeDistanceThreshold);
        return;
    }
    else {
        for (const auto& ii : pIndicesInliers->indices){
            indicesPoints.push_back(static_cast<int>(ii));
        }
        TDO_LOG_DEBUG_FORMAT("plane fitting found %d inliers from %d points, plane distance threshold = %f", indicesPoints.size() % points.size() % planeDistanceThreshold);
        return;
    }
}

std::vector<cv::Point> mathutils::ProjectPoints3DToPoints2D(Eigen::MatrixXf& mPoints3D, camera::CameraBase& camera){
    Eigen::Matrix<float, 2, Eigen::Dynamic> mPoints2D(2, mPoints3D.cols());
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

/**
 *  Return the minimal quaternion that orients sourcedir to targetdir
 *  :param sourcedir: direction of the original vector, 3 values
 *  :param targetdir: new direction, 3 values 
 * 
 */
Eigen::Vector4f mathutils::CreateQuatRotateDirection(const Eigen::Vector3f sourceDir, const Eigen::Vector3f targetDir){
    Eigen::Vector3f rotToDirection = sourceDir.cross(targetDir);
    float fsin = rotToDirection.norm();
    float fcos = sourceDir.dot(targetDir);
    if (fsin > 0){
        return ConvertQuatFromAxisAngle(rotToDirection * (1 / fsin), std::atan2(fsin, fcos));
    }

    if (fcos < 0){  // when sourceDir and targetDir are 180 deg flipped, and fsin is zero
        rotToDirection[0] = 1.0;
        rotToDirection[1] = 0;
        rotToDirection[2] = 0;
        rotToDirection -= sourceDir * sourceDir.dot(rotToDirection);
        if (rotToDirection.norm() < 1e-8){
            rotToDirection[0] = 0;
            rotToDirection[1] = 0;
            rotToDirection[2] = 1.0;
            rotToDirection -= sourceDir * sourceDir.dot(rotToDirection);
        }
        rotToDirection /= rotToDirection.norm();
        return ConvertQuatFromAxisAngle(rotToDirection, std::atan2(fsin, fcos));
    }

    Eigen::Vector4f qIdentityRotation(1.0, 0, 0, 0);
    return qIdentityRotation;
}

/**
 * angle is in radians
**/
Eigen::Vector4f mathutils::ConvertQuatFromAxisAngle(const Eigen::Vector3f axis, const float angle){
    float axisLength = axis.norm();
    if (axisLength <= 0){
        Eigen::Vector4f qIdentityRotation(1.0, 0, 0, 0);
        return qIdentityRotation;
    }
    float sinAngle = std::sin(angle * 0.5) / axisLength;
    float cosAngle = std::cos(angle * 0.5);
    Eigen::Vector4f qRotation(cosAngle, axis[0] * sinAngle, axis[1] * sinAngle, axis[2] * sinAngle);
    return qRotation;
}

/**
 * return 4x4 matrix
**/
Eigen::Matrix4f mathutils::ConvertMatrixFromQuat(const Eigen::Vector4f quat){
    float length2 = quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3];
    float ilength2 = 2.0 / length2;
    float qq1 = ilength2 * quat[1] * quat[1];
    float qq2 = ilength2 * quat[2] * quat[2];
    float qq3 = ilength2 * quat[3] * quat[3];
    Eigen::Matrix4f T;
    T(0, 0) = 1 - qq2 - qq3;
    T(0, 1) = ilength2 * (quat[1] * quat[2] - quat[0] * quat[3]);
    T(0, 2) = ilength2 * (quat[1] * quat[3] + quat[0] * quat[2]);
    T(1, 0) = ilength2 * (quat[1] * quat[2] + quat[0] * quat[3]);
    T(1, 1) = 1 - qq1 - qq3;
    T(1, 2) = ilength2 * (quat[2] * quat[3] - quat[0] * quat[1]);
    T(2, 0) = ilength2 * (quat[1] * quat[3] - quat[0] * quat[2]);
    T(2, 1) = ilength2 * (quat[2] * quat[3] + quat[0] * quat[1]);
    T(2, 2) = 1 - qq1 - qq2;
    return T;
}

std::string mathutils::FillZeros(const std::string& str, const int width)
{
  std::stringstream ss;
  ss << std::setw(width) << std::setfill('0') << str;
  return ss.str();
}

std::vector<size_t> mathutils::GetListOfRandomIndex(const size_t iStart, const size_t iEnd, const size_t numElements) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> distribution(iStart, iEnd - 1);

    std::vector<size_t> randomIndicies;
    randomIndicies.reserve(numElements);

    for (int i = 0; i < numElements; i++) {
        randomIndicies.push_back(distribution(gen));
    }
    return randomIndicies;
}


}  // end of eventobjectslam
