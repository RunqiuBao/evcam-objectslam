#include "mathutils.h"

#include <random>

#include <logging.h>
TDO_LOGGER("objectslam.mathutils")

namespace eventobjectslam {

Mat33_t mathutils::GetRotationMatrixFromVectors(const Vec3_t& vectorA, const Vec3_t& vectorB) {
    Vec3_t v = vectorA.cross(vectorB);
    float c = vectorA.dot(vectorB);
    float s = std::sqrt(1 - c * c);

    Mat33_t vx;
    vx << 0, -v.z(), v.y(),
          v.z(), 0, -v.x(),
          -v.y(), v.x(), 0;

    Mat33_t rotationMatrix = Mat33_t::Identity() + vx + vx * vx * ((1 - c) / (s * s));

    return rotationMatrix;
}

float mathutils::ComputeDistanceFromPlane(const pcl::ModelCoefficients::Ptr plane_coefficients, const pcl::PointXYZ& query_point) {
    // Extract the coefficients
    float a = plane_coefficients->values[0];
    float b = plane_coefficients->values[1];
    float c = plane_coefficients->values[2];
    float d = plane_coefficients->values[3];

    // Compute the distance from the point to the plane
    float distance = std::abs(a * query_point.x + b * query_point.y + c * query_point.z + d) / std::sqrt(a * a + b * b + c * c);

    return distance;
}

void mathutils::EstimatePlaneFromPoints(
    const std::vector<Vec3_t> points,
    const float planeDistanceThreshold,
    pcl::ModelCoefficients::Ptr& planeCoeff,
    pcl::PointIndices::Ptr& pIndicesInliers
){
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
    
    seg.setInputCloud(pPoints);
    seg.segment(*pIndicesInliers, *planeCoeff);
}

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
    pcl::ModelCoefficients::Ptr planeCoeff(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr pIndicesInliers(new pcl::PointIndices);
    EstimatePlaneFromPoints(points, planeDistanceThreshold, planeCoeff, pIndicesInliers);

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

std::vector<cv::Point> mathutils::ProjectPoints3DToPoints2D(const Eigen::MatrixXf& mPoints3D, camera::CameraBase& camera){
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

static bool Triangulate(Eigen::Vector3f &x_c1, Eigen::Vector3f &x_c2,Eigen::Matrix<float,3,4> &Tc1w ,Eigen::Matrix<float,3,4> &Tc2w , Eigen::Vector3f &x3D)
{
    Eigen::Matrix4f A;
    A.block<1,4>(0,0) = x_c1(0) * Tc1w.block<1,4>(2,0) - Tc1w.block<1,4>(0,0);
    A.block<1,4>(1,0) = x_c1(1) * Tc1w.block<1,4>(2,0) - Tc1w.block<1,4>(1,0);
    A.block<1,4>(2,0) = x_c2(0) * Tc2w.block<1,4>(2,0) - Tc2w.block<1,4>(0,0);
    A.block<1,4>(3,0) = x_c2(1) * Tc2w.block<1,4>(2,0) - Tc2w.block<1,4>(1,0);

    Eigen::JacobiSVD<Eigen::Matrix4f> svd(A, Eigen::ComputeFullV);

    Eigen::Vector4f x3Dh = svd.matrixV().col(3);

    if(x3Dh(3)==0)
        return false;

    // Euclidean coordinates
    x3D = x3Dh.head(3)/x3Dh(3);

    return true;
}

int mathutils::CheckRT(const Eigen::Matrix3f& R, const Eigen::Vector3f& t, const std::vector<cv::Point2f>& vKeys1, const std::vector<cv::Point2f>& vKeys2,
            const Eigen::Matrix3f& K, std::vector<cv::Point3f>& vP3D, float th2, std::vector<bool>& vbGood, float& parallax)
{
    // Calibration parameters
    const float fx = K(0,0);
    const float fy = K(1,1);
    const float cx = K(0,2);
    const float cy = K(1,2);

    vbGood = std::vector<bool>(vKeys1.size(), false);
    vP3D.resize(vKeys1.size());

    std::vector<float> vCosParallax;
    vCosParallax.reserve(vKeys1.size());

    // Camera 1 Projection Matrix K[I|0]
    Eigen::Matrix<float,3,4> P1;
    P1.setZero();
    P1.block<3,3>(0,0) = K;

    Eigen::Vector3f O1;
    O1.setZero();

    // Camera 2 Projection Matrix K[R|t]
    Eigen::Matrix<float,3,4> P2;
    P2.block<3,3>(0,0) = R;
    P2.block<3,1>(0,3) = t;
    P2 = K * P2;

    Eigen::Vector3f O2 = -R.transpose() * t;

    int nGood=0;

    for(size_t i=0, iend=vKeys1.size();i<iend;i++)
    {
        const cv::Point2f& kp1 = vKeys1[i];
        const cv::Point2f& kp2 = vKeys2[i];

        Eigen::Vector3f p3dC1;
        Eigen::Vector3f x_p1(kp1.x, kp1.y, 1);
        Eigen::Vector3f x_p2(kp2.x, kp2.y, 1);

        Triangulate(x_p1, x_p2, P1, P2, p3dC1);


        if(!isfinite(p3dC1(0)) || !isfinite(p3dC1(1)) || !isfinite(p3dC1(2)))
        {
            vbGood[i]=false;
            continue;
        }

        // Check parallax
        Eigen::Vector3f normal1 = p3dC1 - O1;
        float dist1 = normal1.norm();

        Eigen::Vector3f normal2 = p3dC1 - O2;
        float dist2 = normal2.norm();

        float cosParallax = normal1.dot(normal2) / (dist1*dist2);

        // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        if(p3dC1(2)<=0 && cosParallax<0.99998)
            continue;
        TDO_LOG_VERBOSE("O1: " << O1 << ", O2: " << O2 << ",\np3dC1: " << p3dC1 << ", cosParallax: " << cosParallax);

        // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        Eigen::Vector3f p3dC2 = R * p3dC1 + t;

        if(p3dC2(2)<=0 && cosParallax<0.99998)
            continue;

        // Check reprojection error in first image
        float im1x, im1y;
        float invZ1 = 1.0/p3dC1(2);
        im1x = fx*p3dC1(0)*invZ1+cx;
        im1y = fy*p3dC1(1)*invZ1+cy;

        float squareError1 = (im1x-kp1.x)*(im1x-kp1.x)+(im1y-kp1.y)*(im1y-kp1.y);

        if(squareError1>th2)
            continue;

        // Check reprojection error in second image
        float im2x, im2y;
        float invZ2 = 1.0/p3dC2(2);
        im2x = fx*p3dC2(0)*invZ2+cx;
        im2y = fy*p3dC2(1)*invZ2+cy;

        float squareError2 = (im2x-kp2.x)*(im2x-kp2.x)+(im2y-kp2.y)*(im2y-kp2.y);

        if(squareError2>th2)
            continue;
        TDO_LOG_VERBOSE("squareError1: " << squareError1);
        TDO_LOG_VERBOSE("squareError2: " << squareError1);

        vCosParallax.push_back(cosParallax);
        vP3D[i] = cv::Point3f(p3dC1(0), p3dC1(1), p3dC1(2));
        nGood++;

        if(cosParallax<0.99998)
            vbGood[i]=true;
    }

    if(nGood>0)
    {
        std::sort(vCosParallax.begin(),vCosParallax.end());

        size_t idx = std::min(50, int(vCosParallax.size()-1));
        parallax = std::acos(vCosParallax[idx])*180/CV_PI;
    }
    else
        parallax=0;

    return nGood;
}

bool mathutils::ReconstructH(
    int numMatchedPoints,
    Eigen::Matrix3f& H21,
    const Eigen::Matrix3f& K,  // camera intrinsics
    Eigen::Matrix4f& T21,
    std::vector<cv::Point2f>& vKeyPts1,
    std::vector<cv::Point2f>& vKeyPts2,
    std::vector<cv::Point3f>& vP3D,
    std::vector<bool>& vbTriangulated,  // need to be same size as vP3D
    float minParallax,
    int minTriangulated
){
    int N=numMatchedPoints;

    // We recover 8 motion hypotheses using the method of Faugeras et al.
    // Motion and structure from motion in a piecewise planar environment.
    // International Journal of Pattern Recognition and Artificial Intelligence, 1988
    Eigen::Matrix3f invK = K.inverse();
    Eigen::Matrix3f A = invK * H21 * K;

    Eigen::JacobiSVD<Eigen::Matrix3f> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f U = svd.matrixU();
    Eigen::Matrix3f V = svd.matrixV();
    Eigen::Matrix3f Vt = V.transpose();
    Eigen::Vector3f w = svd.singularValues();

    float s = U.determinant() * Vt.determinant();

    float d1 = w(0);
    float d2 = w(1);
    float d3 = w(2);

    if(d1/d2<1.00001 || d2/d3<1.00001)
    {
        return false;
    }

    std::vector<Eigen::Matrix3f> vR;
    std::vector<Eigen::Vector3f> vt, vn;
    vR.reserve(8);
    vt.reserve(8);
    vn.reserve(8);

    //n'=[x1 0 x3] 4 posibilities e1=e3=1, e1=1 e3=-1, e1=-1 e3=1, e1=e3=-1
    float aux1 = sqrt((d1*d1-d2*d2)/(d1*d1-d3*d3));
    float aux3 = sqrt((d2*d2-d3*d3)/(d1*d1-d3*d3));
    float x1[] = {aux1,aux1,-aux1,-aux1};
    float x3[] = {aux3,-aux3,aux3,-aux3};

    //case d'=d2
    float aux_stheta = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1+d3)*d2);

    float ctheta = (d2*d2+d1*d3)/((d1+d3)*d2);
    float stheta[] = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta};

    for(int i=0; i<4; i++)
    {
        Eigen::Matrix3f Rp;
        Rp.setZero();
        Rp(0,0) = ctheta;
        Rp(0,2) = -stheta[i];
        Rp(1,1) = 1.f;
        Rp(2,0) = stheta[i];
        Rp(2,2) = ctheta;

        Eigen::Matrix3f R = s*U*Rp*Vt;
        vR.push_back(R);

        Eigen::Vector3f tp;
        tp(0) = x1[i];
        tp(1) = 0;
        tp(2) = -x3[i];
        tp *= d1-d3;

        Eigen::Vector3f t = U*tp;
        vt.push_back(t / t.norm());

        Eigen::Vector3f np;
        np(0) = x1[i];
        np(1) = 0;
        np(2) = x3[i];

        Eigen::Vector3f n = V*np;
        if(n(2) < 0)
            n = -n;
        vn.push_back(n);
    }

    //case d'=-d2
    float aux_sphi = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1-d3)*d2);

    float cphi = (d1*d3-d2*d2)/((d1-d3)*d2);
    float sphi[] = {aux_sphi, -aux_sphi, -aux_sphi, aux_sphi};

    for(int i=0; i<4; i++)
    {
        Eigen::Matrix3f Rp;
        Rp.setZero();
        Rp(0,0) = cphi;
        Rp(0,2) = sphi[i];
        Rp(1,1) = -1;
        Rp(2,0) = sphi[i];
        Rp(2,2) = -cphi;

        Eigen::Matrix3f R = s*U*Rp*Vt;
        vR.push_back(R);

        Eigen::Vector3f tp;
        tp(0) = x1[i];
        tp(1) = 0;
        tp(2) = x3[i];
        tp *= d1+d3;

        Eigen::Vector3f t = U*tp;
        vt.push_back(t / t.norm());

        Eigen::Vector3f np;
        np(0) = x1[i];
        np(1) = 0;
        np(2) = x3[i];

        Eigen::Vector3f n = V*np;
        if(n(2) < 0)
            n = -n;
        vn.push_back(n);
    }


    int bestGood = 0;
    int secondBestGood = 0;
    int bestSolutionIdx = -1;
    float bestParallax = -1;
    float mSigma2 = 1.0;
    std::vector<cv::Point3f> bestP3D;
    std::vector<bool> bestTriangulated;

    // Instead of applying the visibility constraints proposed in the Faugeras' paper (which could fail for points seen with low parallax)
    // We reconstruct all hypotheses and check in terms of triangulated points and parallax
    for (auto& point3D : vP3D) {
        TDO_LOG_VERBOSE("corner3D: " << point3D);
    }
    for(size_t i=0; i<8; i++)
    {
        TDO_LOG_VERBOSE("homography RT candidate " << i << ", R=\n" << vR[i] << ", t=\n" << vt[i]);
        float parallaxi;
        std::vector<cv::Point3f> vP3Di;
        std::vector<bool> vbTriangulatedi;
        int nGood = CheckRT(
            vR[i],
            vt[i],
            vKeyPts1,
            vKeyPts2,
            K,
            vP3Di,
            4.0 * mSigma2,
            vbTriangulatedi,
            parallaxi
        );

        if(nGood>bestGood)
        {
            secondBestGood = bestGood;
            bestGood = nGood;
            bestSolutionIdx = i;
            bestParallax = parallaxi;
            bestP3D = vP3Di;
            bestTriangulated = vbTriangulatedi;
        }
        else if(nGood>secondBestGood)
        {
            secondBestGood = nGood;
        }
    }


    // if(secondBestGood<0.75*bestGood && bestParallax>=minParallax && bestGood>minTriangulated && bestGood>0.9*N)
    if(bestParallax>=minParallax && bestGood>minTriangulated && bestGood>0.9*N)
    {
        T21 = Eigen::Matrix4f::Identity();
        T21.block<3, 3>(0, 0) = vR[bestSolutionIdx];
        T21.col(3).head<3>() = vt[bestSolutionIdx];
        vbTriangulated = bestTriangulated;
        TDO_LOG_DEBUG("Got homography solution:= \n" << T21);

        return true;
    }
    else{
        TDO_LOG_DEBUG("decompose homography failed due to: \nsecondBestGood " << secondBestGood << ", 0.75*bestGood: " << 0.75*bestGood << "\nbestParallax: " << bestParallax << ", minParallax: " << minParallax << "\nbestGood: " << bestGood << ", minTriangulated: " << minTriangulated << "\nbestGood: " << bestGood << ", 0.9*N: " << 0.9*N);
    }

    return false;
}

bool mathutils::TrackWithHomography(
    std::vector<cv::Point2f>& currPoints2D,
    std::vector<cv::Point2f>& refPoints2D,
    std::vector<cv::Point3f>& currPoints3D,
    const Eigen::Matrix3f& cameraMatrix,
    Mat44_t& currFrameInRefKeyFrame
){
    std::vector<uint8_t> mask;  // Note: filter outliers by ransac
    cv::Mat H = cv::findHomography(currPoints2D, refPoints2D, 0, 3, mask);

    // reprojection verify
    std::vector<cv::Point2f> refPoints2DReproj(refPoints2D.size());
    cv::perspectiveTransform(currPoints2D, refPoints2DReproj, H);
    float sumError = 0;
    TDO_LOG_DEBUG("H: \n" << H);
    for (size_t indexPoint=0; indexPoint < refPoints2D.size(); indexPoint++){
        sumError += cv::norm(refPoints2D[indexPoint] - refPoints2DReproj[indexPoint]);
    }
    TDO_LOG_DEBUG("avg reproj. error = " << sumError / refPoints2D.size());

    // decompose rotation and translation from H matrix
    Eigen::Matrix3d mH;
    mH << H.at<double>(0, 0), H.at<double>(0, 1), H.at<double>(0, 2),
          H.at<double>(1, 0), H.at<double>(1, 1), H.at<double>(1, 2),
          H.at<double>(2, 0), H.at<double>(2, 1), H.at<double>(2, 2);
    Eigen::Matrix3f mfH = mH.cast<float>();
    std::vector<bool> vbTriangulated;
    float minParallax = 0.1;  // in degree.
    int minTriangulated = 3;
    bool isSuccess = ReconstructH(
        4,
        mfH,
        cameraMatrix,  // camera intrinsics
        currFrameInRefKeyFrame,
        currPoints2D,
        refPoints2D,
        currPoints3D,
        vbTriangulated,
        minParallax,
        minTriangulated
    );

    TDO_LOG_DEBUG("tracke with homography " << (isSuccess?"succeeded!":"failed..."));

    return isSuccess;
}

void mathutils::RestoreTranslationScale(
    Eigen::Matrix4f& T21,
    std::vector<cv::Point2f>& c1Points2D,
    std::vector<cv::Point2f>& c2Points2D,
    std::vector<cv::Point3f>& c1Points3D,
    const Eigen::Matrix3f& K
){
    // camera 1 projection matrix K[I|0]
    Eigen::Matrix<float, 3, 4> P1;
    P1.setZero();
    P1.block<3, 3>(0, 0) = K;

    // camera 2 projection matrix K[R|t]
    Eigen::Matrix<float, 3, 4> P2;
    P2.block<3, 3>(0, 0) = T21.block<3, 3>(0, 0);
    P2.block<3, 1>(0, 3) = T21.col(3).head<3>();
    P2 = K * P2;

    // real depth
    float avgDepthInC1 = 0;
    for (auto& point3dInC1 : c1Points3D){
        avgDepthInC1 += point3dInC1.z;
    }
    avgDepthInC1 /= c1Points3D.size();

    // depth triangulated by T21
    float avgDepthInC1_triangulated = 0;
    for (size_t ii=0; ii < c1Points2D.size(); ii++){
        const cv::Point2f& kpC1 = c1Points2D[ii];
        const cv::Point2f& kpC2 = c2Points2D[ii];

        Eigen::Vector3f p3dC1_triangulated;
        Eigen::Vector3f x_kpC1(kpC1.x, kpC1.y, 1);
        Eigen::Vector3f x_kpC2(kpC2.x, kpC2.y, 1);
        Triangulate(x_kpC1, x_kpC2, P1, P2, p3dC1_triangulated);
        avgDepthInC1_triangulated += p3dC1_triangulated[2];
    }
    avgDepthInC1_triangulated /= c1Points2D.size();

    TDO_LOG_DEBUG("real depth: " << avgDepthInC1 << ", depth triangulated by T21: " << avgDepthInC1_triangulated);
    T21.col(3).head<3>() *= avgDepthInC1 / avgDepthInC1_triangulated;
}



}  // end of eventobjectslam
