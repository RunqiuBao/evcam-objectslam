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


}  // end of eventobjectslam
