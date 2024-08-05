#include "mathutils.h"
#include "frametracker.h"
#include "mapdatabase.h"
#include "landmark.h"

#include <filesystem>
#include <algorithm>
#include <stdexcept>

#include <opencv2/tracking.hpp>
#include <opencv2/tracking/tracking_legacy.hpp>
#include <opencv2/imgproc.hpp>

// #include <unsupported/Eigen/MatrixFunctions>

#include <logging.h>
TDO_LOGGER("eventobjectslam.frametracker")

namespace eventobjectslam {

void SaveCvmatForDebug(cv::Mat& debugImg_in, const std::string debugId){
    cv::Mat debugImg_int;
    cv::Mat debugImg = cv::abs(debugImg_in.clone());
    cv::Scalar sum = cv::sum(debugImg);
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(debugImg, &minVal, &maxVal, &minLoc, &maxLoc);
    debugImg = debugImg * 255 / maxVal;
    debugImg.convertTo(debugImg_int, CV_32S);
    cv::imwrite("/home/runqiu/tmptmp/debug_" + debugId + ".png", debugImg_int);
}

static Mat44_d se3Exp(Vec6_d twist){
    // Mat44_d incrementPose;
    // incrementPose << 0, -twist(5), twist(4), twist(0),
    //                 twist(5), 0, -twist(3), twist(1),
    //                 -twist(4), twist(3), 0, twist(3),
    //                 0, 0, 0, 0;
    // Mat44_d expIncrePose = incrementPose.exp();

    arma::mat incrementPoseArma(4, 4);  // Note: eigen unsupported broke g2o
    incrementPoseArma(0, 0) = 0.0;
    incrementPoseArma(0, 1) = -twist(5);
    incrementPoseArma(0, 2) = twist(4);
    incrementPoseArma(0, 3) = twist(0);
    incrementPoseArma(1, 0) = twist(5);
    incrementPoseArma(1, 1) = 0.0;
    incrementPoseArma(1, 2) = -twist(3);
    incrementPoseArma(1, 3) = twist(1);
    incrementPoseArma(2, 0) = -twist(4);
    incrementPoseArma(2, 1) = twist(3);
    incrementPoseArma(2, 2) = 0.0;
    incrementPoseArma(2, 3) = twist(3);
    incrementPoseArma(3, 0) = 0.0;
    incrementPoseArma(3, 1) = 0.0;
    incrementPoseArma(3, 2) = 0.0;
    incrementPoseArma(3, 3) = 0.0;
    // Compute the matrix exponential
    arma::mat expIncrePoseArma = arma::expmat(incrementPoseArma);
    Mat44_d expIncrePose = Mat44_d::Identity();
    expIncrePose << expIncrePoseArma(0, 0), expIncrePoseArma(0, 1), expIncrePoseArma(0, 2), expIncrePoseArma(0, 3),
                        expIncrePoseArma(1, 0), expIncrePoseArma(1, 1), expIncrePoseArma(1, 2), expIncrePoseArma(1, 3),
                        expIncrePoseArma(2, 0), expIncrePoseArma(2, 1), expIncrePoseArma(2, 2), expIncrePoseArma(2, 3),
                        expIncrePoseArma(3, 0), expIncrePoseArma(3, 1), expIncrePoseArma(3, 2), expIncrePoseArma(3, 3);
    return expIncrePose;
}

static Vec6_d se3Log(Mat44_d incrementPose){
    Vec6_d twist = Vec6_d::Zero();
    if (incrementPose.isIdentity()){
        return twist;
    }
    // Eigen::MatrixXd incrementPoseLog = incrementPose.log();
    // twist << incrementPoseLog(0, 3), incrementPoseLog(1, 3), incrementPoseLog(2, 3), incrementPoseLog(2, 1), incrementPoseLog(0, 2), incrementPoseLog(1, 0);

    arma::mat incrementPoseArma(4, 4);
    incrementPoseArma(0, 0) = incrementPose(0, 0);
    incrementPoseArma(0, 1) = incrementPose(0, 1);
    incrementPoseArma(0, 2) = incrementPose(0, 2);
    incrementPoseArma(0, 3) = incrementPose(0, 3);
    incrementPoseArma(1, 0) = incrementPose(1, 0);
    incrementPoseArma(1, 1) = incrementPose(1, 1);
    incrementPoseArma(1, 2) = incrementPose(1, 2);
    incrementPoseArma(1, 3) = incrementPose(1, 3);
    incrementPoseArma(2, 0) = incrementPose(2, 0);
    incrementPoseArma(2, 1) = incrementPose(2, 1);
    incrementPoseArma(2, 2) = incrementPose(2, 2);
    incrementPoseArma(2, 3) = incrementPose(2, 3);
    incrementPoseArma(3, 0) = incrementPose(3, 0);
    incrementPoseArma(3, 1) = incrementPose(3, 1);
    incrementPoseArma(3, 2) = incrementPose(3, 2);
    incrementPoseArma(3, 3) = incrementPose(3, 3);
    arma::cx_mat incrementPoseLogArma = arma::logmat(incrementPoseArma);
    arma::mat incrementPoseLogRealArma = arma::real(incrementPoseLogArma);
    twist << incrementPoseLogRealArma(0, 3), incrementPoseLogRealArma(1, 3), incrementPoseLogRealArma(2, 3), incrementPoseLogRealArma(2, 1), incrementPoseLogRealArma(0, 2), incrementPoseLogRealArma(1, 0);
    return twist;
}

static void DownScale(cv::Mat& featImage, cv::Mat& depthImage, Mat33_t& kk, const float downScaleFactor){
    Mat33_t kkLevel;
    kkLevel << kk(0, 0) / downScaleFactor, 0, kk(0, 2) / downScaleFactor,
                0, kk(1, 1) / downScaleFactor, kk(1, 2) / downScaleFactor,
                0, 0, 1;
    kk = kkLevel;
    cv::resize(featImage, featImage, cv::Size(), 1/ downScaleFactor, 1 / downScaleFactor, cv::INTER_AREA);
    cv::resize(depthImage, depthImage, cv::Size(), 1 / downScaleFactor, 1 / downScaleFactor, cv::INTER_NEAREST);
}

void PoseOptimizer::DrawRectangleWithInverseDistance(cv::Mat& featmap, cv::Rect rect, const bool isNormalize){
    cv::Point center(rect.x + rect.width / 2, rect.y + rect.height / 2);
    float maxDistance = std::sqrt(std::pow(rect.x - center.x, 2) + std::pow(rect.y - center.y, 2));
    for (int y = rect.y; y < rect.y + rect.height; y++){
        for (int x = rect.x; x < rect.x + rect.width; x++){
            float distance = std::sqrt(std::pow(x - center.x, 2) + std::pow(y - center.y, 2));
            featmap.at<float>(y, x) = maxDistance - distance;
        }
    }
    if (isNormalize){
        cv::normalize(featmap, featmap, 0, 1, cv::NORM_MINMAX);
    }
}

void PoseOptimizer::PrepareDepthAndFeatureMapFromBboxes(
    std::vector<TwoDBoundingBox>& leftBboxes,
    std::vector<TwoDBoundingBox>& rightBboxes,
    const int imageWidth,
    const int imageHeight,
    cv::Mat& featureMap,
    cv::Mat& depthMap
){
    featureMap = cv::Mat::zeros(imageHeight, imageWidth, CV_32FC1);
    depthMap = cv::Mat::zeros(imageHeight, imageWidth, CV_32FC1);
    for (size_t ii=0; ii < leftBboxes.size(); ii++){
        auto& leftbbox = leftBboxes[ii];
        auto& rightbbox = rightBboxes[ii];
        cv::Mat mask = cv::Mat::zeros(featureMap.size(), CV_8UC1);
        cv::Rect cvBbox = cv::Rect(
            static_cast<int>(leftbbox._centerX - leftbbox._bWidth / 2),
            static_cast<int>(leftbbox._centerY - leftbbox._bHeight / 2),
            static_cast<int>(leftbbox._bWidth),
            static_cast<int>(leftbbox._bHeight));
        cv::rectangle(mask, cvBbox, cv::Scalar(255), cv::FILLED);

        float disparity = leftbbox._centerX - rightbbox._centerX;
        float depth = _kk(0, 0) * _baseline / disparity;
        depthMap.setTo(depth, mask);

        this->DrawRectangleWithInverseDistance(featureMap, cvBbox, false);
    }

    cv::Mat residual_int;
    cv::Mat residual_debug = cv::abs(featureMap.clone());
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(residual_debug, &minVal, &maxVal, &minLoc, &maxLoc);
    residual_debug = residual_debug * 255 / maxVal;
    residual_debug.convertTo(residual_int, CV_32S);
}

void PoseOptimizer::deriveAnalytic(
    const cv::Mat& refDepth,
    const cv::Mat& refImage,
    const std::vector<TwoDBoundingBox>& bboxesRef,
    const cv::Mat& currImage,
    const Vec6_d& xi,
    const Mat33_d kk,
    const int scaleLevel,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& outJac,
    Eigen::VectorXf& outResidual
){
    Mat44_d T = se3Exp(xi);
    Mat33_d R = T.block<3, 3>(0, 0);
    Vec3_d t = T.col(3).head<3>();
    Mat33_d RKinv = R * kk.inverse();

    int imageWidth = refImage.cols;
    int imageHeight = refImage.rows;
    // record the projection of 3d points transformed from ref to curr.
    cv::Mat xImg(imageHeight, imageWidth, CV_32F, cv::Scalar(-10.0));
    cv::Mat yImg(imageHeight, imageWidth, CV_32F, cv::Scalar(-10.0));
    // record the 3d points transformed from ref to curr.
    cv::Mat xp(imageHeight, imageWidth, CV_32F, std::numeric_limits<float>::quiet_NaN());
    cv::Mat yp(imageHeight, imageWidth, CV_32F, std::numeric_limits<float>::quiet_NaN());
    cv::Mat zp(imageHeight, imageWidth, CV_32F, std::numeric_limits<float>::quiet_NaN());

    for (size_t ii=0; ii < bboxesRef.size(); ii++){
        auto& bbox = bboxesRef[ii];
        cv::Rect rect = cv::Rect(
            static_cast<int>((bbox._centerX - bbox._bWidth / 2) / std::pow(2, scaleLevel)),
            static_cast<int>((bbox._centerY - bbox._bHeight / 2) / std::pow(2, scaleLevel)),
            static_cast<int>(bbox._bWidth / std::pow(2, scaleLevel)),
            static_cast<int>(bbox._bHeight / std::pow(2, scaleLevel)));
        for (int y = rect.y; y < rect.y + rect.height; y++){
            for (int x = rect.x; x < rect.x + rect.width; x++){
                if (x < 0 || x >= imageWidth || y < 0 || y >= imageHeight){
                    continue;
                }
                Vec3_d p(x, y, 1);
                p = p * (double)refDepth.at<float>(y, x);
                Vec3_d pTrans = RKinv * p + t;

                // if point is valid (depth > 0), project and save result.
                if (pTrans(2) > 0 && refDepth.at<float>(y, x) > 0){
                    Vec3_d pTransProj = kk * pTrans;
                    xImg.at<float>(y, x) = pTransProj(0) / pTransProj(2);
                    yImg.at<float>(y, x) = pTransProj(1) / pTransProj(2);

                    xp.at<float>(y, x) = pTrans(0);
                    yp.at<float>(y, x) = pTrans(1);
                    zp.at<float>(y, x) = pTrans(2);
                }
            }
        }
    }

    // calculate actual derivative
    std::cout << "baodebug" << std::endl;
    cv::Mat dxI(imageHeight, imageWidth, CV_32F, std::numeric_limits<float>::quiet_NaN());
    cv::Mat dyI(imageHeight, imageWidth, CV_32F, std::numeric_limits<float>::quiet_NaN());

    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mCurrImg(const_cast<float*>(currImage.ptr<float>()), imageHeight, imageWidth);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mDyI_r2endm1 = 0.5 * (mCurrImg.block(2, 0, imageHeight - 2, imageWidth) - mCurrImg.block(0, 0, imageHeight - 2, imageWidth));
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mDxI_c2endm1 = 0.5 * (mCurrImg.block(0, 2, imageHeight, imageWidth - 2) - mCurrImg.block(0, 0, imageHeight, imageWidth - 2));
    cv::Mat dyI_r2endm1(imageHeight - 2, imageWidth, CV_32F, mDyI_r2endm1.data());
    cv::Mat dxI_c2endm1(imageHeight, imageWidth - 2, CV_32F, mDxI_c2endm1.data());

    cv::Point topLeft_r2endm1(0, 1);
    cv::Point bottomRight_r2endm1(imageWidth, imageHeight - 1);
    cv::Rect roi_r2endm1(topLeft_r2endm1, bottomRight_r2endm1);
    dyI_r2endm1.copyTo(dyI(roi_r2endm1));

    cv::Point topLeft_c2endm1(1, 0);
    cv::Point bottomRight_c2endm1(imageWidth - 1, imageHeight);
    cv::Rect roi_c2endm1(topLeft_c2endm1, bottomRight_c2endm1);
    dxI_c2endm1.copyTo(dxI(roi_c2endm1));

    cv::Mat dxI_warp;
    // Perform the remapping
    cv::remap(dxI, dxI_warp, xImg, yImg, cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
    Eigen::VectorXf mIxfx = kk(0, 0) * Eigen::Map<Eigen::VectorXf>(const_cast<float*>(dxI_warp.ptr<float>()), imageHeight * imageWidth);
    cv::Mat dyI_warp;
    cv::remap(dyI, dyI_warp, xImg, yImg, cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
    Eigen::VectorXf mIyfy = kk(1, 1) * Eigen::Map<Eigen::VectorXf>(const_cast<float*>(dyI_warp.ptr<float>()), imageHeight * imageWidth);

    // flatten xp, yp, zp
    Eigen::VectorXf mXp = Eigen::Map<Eigen::VectorXf>(const_cast<float*>(xp.ptr<float>()), imageHeight * imageWidth);
    Eigen::VectorXf mYp = Eigen::Map<Eigen::VectorXf>(const_cast<float*>(yp.ptr<float>()), imageHeight * imageWidth);
    Eigen::VectorXf mZp = Eigen::Map<Eigen::VectorXf>(const_cast<float*>(zp.ptr<float>()), imageHeight * imageWidth);

    // jacobian matrix
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mJac(imageHeight * imageWidth, 6);
    mJac.setZero();
    mJac.block(0, 0, imageHeight * imageWidth, 1) = mIxfx.array() / mZp.array();
    mJac.block(0, 1, imageHeight * imageWidth, 1) = mIyfy.array() / mZp.array();
    mJac.block(0, 2, imageHeight * imageWidth, 1) = - (mIxfx.array() * mXp.array() + mIyfy.array() * mYp.array()) / (mZp.array() * mZp.array());
    mJac.block(0, 3, imageHeight * imageWidth, 1) = -(mIxfx.array() * mXp.array() * mYp.array()) / (mZp.array() * mZp.array()) - mIyfy.array() * (1 + (mYp.array() / mZp.array()).square());
    mJac.block(0, 4, imageHeight * imageWidth, 1) = mIxfx.array() * (1 + (mXp.array() / mZp.array()).square()) + (mIyfy.array() * mXp.array() * mYp.array()) / (mZp.array() * mZp.array());
    mJac.block(0, 5, imageHeight * imageWidth, 1) = (-mIxfx.array() * mYp.array() + mIyfy.array() * mXp.array()) / mZp.array();
    outJac = std::move(mJac);

    // plot residual image for debug
    cv::Mat currImage_warp;
    cv::remap(currImage, currImage_warp, xImg, yImg, cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(0.));
    cv::Mat residual = (currImage_warp - refImage);
    outResidual = Eigen::Map<Eigen::VectorXf>(const_cast<float*>(residual.ptr<float>()), imageHeight * imageWidth);
    
    cv::Mat residual_int;
    cv::Mat residual_debug = cv::abs(residual.clone());
    cv::Scalar sum = cv::sum(residual_debug);
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(residual_debug, &minVal, &maxVal, &minLoc, &maxLoc);
    residual_debug = residual_debug * 255 / maxVal;
    residual_debug.convertTo(residual_int, CV_32S);
    cv::imwrite("/home/runqiu/tmptmp/debug.png", residual_int);
}

void PoseOptimizer::EstimatePose(
    const cv::Mat& refDepth,
    const cv::Mat& refImage,
    const std::vector<TwoDBoundingBox>& bboxesRef,
    const cv::Mat& currDepth,
    const cv::Mat& currImage,
    Mat44_d& currInRefTransform
){
    Vec6_d xi = Vec6_d::Zero();

    for (size_t iiLevel=_numLevel; iiLevel > 0; iiLevel--){
        // downscale inputs. from rough to fine.
        Mat33_t kk_level = _kk;
        Mat33_t kk_clone = _kk;
        cv::Mat refDepth_clone = refDepth.clone();
        cv::Mat refImage_clone = refImage.clone();
        cv::Mat currDepth_clone = currDepth.clone();
        cv::Mat currImage_clone = currImage.clone();
        DownScale(refDepth_clone, refImage_clone, kk_level, std::pow(2, iiLevel));
        DownScale(currDepth_clone, currImage_clone, kk_clone, std::pow(2, iiLevel));

        float errorLast = std::numeric_limits<float>::max();
        Vec6_d xi_last = Vec6_d::Zero();
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> outJac;
        Eigen::VectorXf outResidual;
        bool haveDecreased = false;
        for (size_t iter = 0; iter < 10; iter++){
            deriveAnalytic(
                refDepth_clone,
                refImage_clone,
                bboxesRef,
                currImage_clone,
                xi,
                kk_level.cast<double>(),
                iiLevel,
                outJac,
                outResidual
            );
            // remove nan values
            outJac = outJac.unaryExpr([](float v) { return std::isnan(v) ? 0.0f : v; });
            outResidual = outResidual.unaryExpr([](float v) { return std::isnan(v) ? 0.0f : v; });
            
            Eigen::Matrix<float, 6, 6> coeffMat = -(outJac.transpose() * outJac);
            Eigen::Vector<float, 6> constMat = outJac.transpose() * outResidual;
            Vec6_d upd = (coeffMat.ldlt().solve(constMat)).cast<double>();

            xi = se3Log(se3Exp(upd) * se3Exp(xi));

            float avgError = (outResidual.array() * outResidual.array()).mean();
            if (avgError == 0){
                break;  // Note: bboxesRef is empty
            }
            if (avgError / errorLast > 0.99995 && haveDecreased){
                xi = xi_last;
                std::cout << "level: " << iiLevel << ", iter: " << iter <<  ", estimate pose converge early. avgError: " << avgError << std::endl;
                break;
            }
            else if (avgError < errorLast && errorLast < (std::numeric_limits<float>::max() / 2)){
                haveDecreased = true;  // Note: have got good estimate at least once.
            }
            std::cout << "level: " << iiLevel << ", iter: " << iter <<  ", avgError: " << avgError << std::endl;
            errorLast = avgError;
            xi_last = xi;
        }

    }
    currInRefTransform = se3Exp(xi);
    std::cout << "currInRefTransform:\n" << currInRefTransform << std::endl;

}

static void TrackWithHomography(
    const std::vector<cv::Point2f>& srcPoints2D,
    const std::vector<cv::Point2f>& dstPoints2D,
    const cv::Mat& cameraMatrix,
    Mat44_t& currFrameInRefKeyFrame
){
    const float cameraFloorHeight = 0.05;
    std::vector<uint8_t> mask;  // Note: filter outliers by ransac
    cv::Mat H = cv::findHomography(srcPoints2D, dstPoints2D, 0, 3, mask);

    // reprojection verify
    std::vector<cv::Point2f> dstPoints2DReproj(dstPoints2D.size());
    cv::perspectiveTransform(srcPoints2D, dstPoints2DReproj, H);
    float sumError = 0;
    TDO_LOG_DEBUG("H: \n" << H);
    for (size_t indexPoint=0; indexPoint < dstPoints2D.size(); indexPoint++){
        sumError += cv::norm(dstPoints2D[indexPoint] - dstPoints2DReproj[indexPoint]);
    }
    TDO_LOG_DEBUG("avg reproj. error = " << sumError / dstPoints2D.size());

    std::vector<cv::Mat> Rs, ts, Ns;
    cv::decomposeHomographyMat(H, cameraMatrix, Rs, ts, Ns);

    std::vector<uint8_t> possibles;
    cv::filterHomographyDecompByVisibleRefpoints(Rs, Ns, srcPoints2D, dstPoints2D, possibles, mask);
    TDO_LOG_DEBUG_FORMAT("found %d solutions.", possibles.size());
    for (uint8_t ii : possibles){
        TDO_LOG_DEBUG("Rs: " << Rs[ii]);
        TDO_LOG_DEBUG("ts: " << cameraFloorHeight * ts[ii]);
    }
    TDO_LOG_DEBUG("exit");
}

static void TrackWithPnP(
    const std::vector<cv::Point3f>& objectPoints,
    const std::vector<cv::Point2f>& imagePoints,
    const cv::Mat& cameraMatrix,
    const cv::Mat& distCoeffs,
    const float maxPoseError,  // Note: assume camera pose should be close to the origin point. if over this error, use ransacpnp instead.
    Mat44_t& currentFrameInRefKeyFrame
){
    // Rotation vector and translation vector
    cv::Mat rvec, tvec;
    cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, false, cv::SOLVEPNP_EPNP);
    TDO_LOG_DEBUG("rvec: " << rvec << ", tvec: " << tvec);
    if (std::sqrt(tvec.at<double>(0)*tvec.at<double>(0) + tvec.at<double>(1)*tvec.at<double>(1) + tvec.at<double>(2)*tvec.at<double>(2)) > maxPoseError){
        cv::solvePnPRansac(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, false, 20, 3.0);  // iterationsCount = 20, 	reprojectionError = 3.0
    }
    TDO_LOG_DEBUG("(ransac) rvec: " << rvec << ", tvec: " << tvec);
    cv::Mat rotationMatrix;
    cv::Rodrigues(rvec, rotationMatrix);
    Mat44_t refKeyFrameInCurrentFrame = Eigen::Matrix4f::Identity();
    for (int i=0; i < 3; i++){
        for (int j=0; j<3; j++){
            refKeyFrameInCurrentFrame(i, j) = rotationMatrix.at<double>(i, j);
        }
    }
    refKeyFrameInCurrentFrame(0, 3) = tvec.at<double>(0);
    refKeyFrameInCurrentFrame(1, 3) = tvec.at<double>(1);
    refKeyFrameInCurrentFrame(2, 3) = tvec.at<double>(2);
    currentFrameInRefKeyFrame = refKeyFrameInCurrentFrame.inverse();
}

bool FrameTracker::DoRelocalizeFromMap(Frame& currentFrame, const Frame& lastFrame, std::shared_ptr<MapDataBase> pMapDb, Mat44_t& velocity, const bool isDebug){
    float maxPoseError = 0.2;
    float minIoUToReject = 0.2;

    std::vector<std::shared_ptr<LandMark>> visibleLandmarks = pMapDb->GetVisibleLandmarks(_pRefKeyframe);
    if (visibleLandmarks.size() < 4) {
        velocity = Eigen::Matrix4f::Identity();
        currentFrame.SetPose(lastFrame.GetPose() * velocity);
        TDO_LOG_DEBUG("relocalization failed. not enough visible landmarks.");
        return false;
    }
    std::vector<int> indicesCorrespondingDetecton;
    indicesCorrespondingDetecton.reserve(visibleLandmarks.size());
    int imgHeight = (*_pRefKeyframe->_pCamera)._rows;
    int imgWidth = (*_pRefKeyframe->_pCamera)._cols;
    cv::Mat displayLdms(imgHeight, imgWidth, CV_8UC1, cv::Scalar(0));
    cv::Mat displayDetections(imgHeight, imgWidth, CV_8UC1, cv::Scalar(0));
    camera::CameraBase& camInstance = (*_pRefKeyframe->_pCamera);
    for (int indexVisibleLandmark=0; indexVisibleLandmark < visibleLandmarks.size(); indexVisibleLandmark++){
        std::shared_ptr<LandMark> pOneLandmark = visibleLandmarks[indexVisibleLandmark];
        Eigen::MatrixXf transformedVerticesInWorld = mathutils::TransformPoints<Eigen::MatrixXf>(pOneLandmark->GetLandmarkPoseInWorld(), pOneLandmark->_vertices3DInLandmark);
        Eigen::MatrixXf transformedVerticesInCamera = mathutils::TransformPoints<Eigen::MatrixXf>((_pRefKeyframe->GetKeyframePoseInWorld()).inverse(), transformedVerticesInWorld);
        std::vector<cv::Point> oneLandmarkPoints2D = mathutils::ProjectPoints3DToPoints2D(transformedVerticesInCamera, camInstance);
        cv::Mat oneLandmarkPoseMask = mathutils::Draw2DHullMaskFrom2DPointsSet(oneLandmarkPoints2D, imgHeight, imgWidth);
        if (isDebug){
            cv::bitwise_or(oneLandmarkPoseMask, displayLdms, displayLdms);
        }
        size_t countDetection = 0;
        int indexLargestOverlap = -1;
        float largestIoU = -1;
        for (ThreeDDetection currentDetection : currentFrame._threeDDetections){
            std::vector<cv::Point> points2DCV = mathutils::ProjectPoints3DToPoints2D(currentDetection._vertices3DInRefFrame, camInstance);
            cv::Mat currentDetectionPoseMask = mathutils::Draw2DHullMaskFrom2DPointsSet(points2DCV, imgHeight, imgWidth);
            cv::Mat overlaps, unions;
            cv::bitwise_and(oneLandmarkPoseMask, currentDetectionPoseMask, overlaps);
            cv::bitwise_or(oneLandmarkPoseMask, currentDetectionPoseMask, unions);
            cv::Scalar sumOverlaps = cv::sum(overlaps);
            cv::Scalar sumUnions = cv::sum(unions);
            if ((sumOverlaps[0] / sumUnions[0]) > largestIoU && (sumOverlaps[0] / sumUnions[0]) > minIoUToReject){
                indexLargestOverlap = countDetection;
                largestIoU = (sumOverlaps[0] / sumUnions[0]);
            }
            countDetection++;
            if (isDebug && indexVisibleLandmark == 0){
                cv::bitwise_or(currentDetectionPoseMask, displayDetections, displayDetections);
            }
        }
        indicesCorrespondingDetecton.push_back(indexLargestOverlap);
    }
    if (isDebug){
        std::vector<cv::Mat> channels;
        cv::Mat zeroChannel(imgHeight, imgWidth, CV_8UC1, cv::Scalar(0));
        cv::Scalar sum = cv::sum(displayLdms);
        cv::Scalar sum2 = cv::sum(displayDetections);
        TDO_LOG_DEBUG("displayRefObjects sum: " << sum[0]);
        TDO_LOG_DEBUG("displayDetections sum: " << sum2[0]);
        channels.push_back(displayLdms * 255);
        channels.push_back(displayDetections * 255);
        channels.push_back(zeroChannel * 255);
        cv::Mat debugRelocal;
        cv::merge(channels, debugRelocal);
        std::filesystem::path debugRelocalPath = _sStereoSequencePathForDebug;
        debugRelocalPath.append("debugRelocalization/");
        if (!std::filesystem::exists(debugRelocalPath) && !std::filesystem::create_directory(debugRelocalPath)){
            TDO_LOG_ERROR_FORMAT("Failed to create the folder: %s", debugRelocalPath.string());
            throw std::runtime_error("Debug");
        }
        debugRelocalPath.append(mathutils::FillZeros(std::to_string(static_cast<int>(currentFrame._timestamp)), 6) + ".png");
        cv::imwrite(debugRelocalPath.string() , debugRelocal);
    }

    // 3D object points in world coordinates
    std::vector<cv::Point3f> objectPoints;
    // Populate objectPoints with the corresponding 3D points from the landmark.
    // 2D image points in image coordinates
    std::vector<cv::Point2f> imagePoints;
    // Populate imagePoints with the corresponding 2D points from the detections in current frame.
    size_t indexLandmark = 0;
    for (int indexCorrespondingDetection : indicesCorrespondingDetecton){
        if (indexCorrespondingDetection < 0){
            indexLandmark++;
            continue;
        }
        // object center
        cv::Point3f point3D(
            visibleLandmarks[indexLandmark]->GetLandmarkPoseInWorld()(0, 3),
            visibleLandmarks[indexLandmark]->GetLandmarkPoseInWorld()(1, 3),
            visibleLandmarks[indexLandmark]->GetLandmarkPoseInWorld()(2, 3)
        );
        objectPoints.push_back(point3D);
        cv::Point2f point2D(
            (*currentFrame._matchedLeftCamDetections[indexCorrespondingDetection])._centerX,
            (*currentFrame._matchedLeftCamDetections[indexCorrespondingDetection])._centerY
        );
        imagePoints.push_back(point2D);
        indexLandmark++;
    }

    // Estimate current frame pose using PnP
    if (objectPoints.size() < 4){
        // relocal failed
        velocity = Eigen::Matrix4f::Identity();
        currentFrame.SetPose(lastFrame.GetPose() * velocity);
        TDO_LOG_DEBUG("relocalization failed. not updating camera pose.");
        // not updating cameraInWorld
        return false;
    }
    else{
        // Camera intrinsic matrix (3x3)
        cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
        cameraMatrix.at<double>(0, 0) = static_cast<double>(camInstance._kk(0, 0));
        cameraMatrix.at<double>(0, 2) = static_cast<double>(camInstance._kk(0, 2));
        cameraMatrix.at<double>(1, 1) = static_cast<double>(camInstance._kk(1, 1));
        cameraMatrix.at<double>(1, 2) = static_cast<double>(camInstance._kk(1, 2));
        // Set the appropriate values for the cameraMatrix
        // Distortion coefficients
        cv::Mat distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
        // Set the appropriate values for the distCoeffs
        Mat44_t currentFrameInWorld, currentFrameInRefKeyFrame;
        TrackWithPnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, maxPoseError, currentFrameInWorld);
        TDO_LOG_DEBUG("relocalization result current Frame in world: \n" << currentFrameInWorld);
        currentFrameInRefKeyFrame = (_pRefKeyframe->GetKeyframePoseInWorld()).inverse() * currentFrameInWorld;
        velocity = lastFrame.GetPose().inverse() * currentFrameInRefKeyFrame;
        if (
            velocity.block(0, 3, 3, 1).norm() > maxPoseError
        ){
            // track failed
            velocity = Eigen::Matrix4f::Identity();
            currentFrame.SetPose(lastFrame.GetPose() * velocity);
            TDO_LOG_DEBUG("relocalization failed. not updating camera pose.");
            // not updating cameraInWorld
            return false;
        }
        else{
            currentFrame.SetPose(currentFrameInRefKeyFrame);
            currentFrame._isTracked = true;
            TDO_LOG_CRITICAL("relocalization succeeded. currentFrameInLast:\n" << velocity);
            return true;
        }
    }

}

bool FrameTracker::DoMotionBasedTrack(Frame& currentFrame, const Frame& lastFrame, Mat44_t& velocity, const bool isDebug) const{
    float minIoUToReject = 0.2;
    float maxPoseError = 0.2;
    float maxRotationAngleDeg = 7.0;

    std::vector<std::shared_ptr<RefObject>> refObjects = _pRefKeyframe->_refObjects;
    // project 3d refObjects and 3d detections to current camera pose and find correspondences.
    std::vector<int> indicesCorrespondingDetecton;
    indicesCorrespondingDetecton.reserve(refObjects.size());
    size_t countRefObject = 0;
    camera::CameraBase myStereoCamera = *_camera;
    cv::Mat displayRefObjects(myStereoCamera._rows, myStereoCamera._cols, CV_8UC1, cv::Scalar(0));
    cv::Mat displayDetections(myStereoCamera._rows, myStereoCamera._cols, CV_8UC1, cv::Scalar(0));
    // for projection based dense alignment
    std::vector<TwoDBoundingBox> leftCamProjections_ref, rightCamProjections_ref, leftCamProjections, rightCamProjections;
    for (std::shared_ptr<RefObject> refObject : refObjects){
        Eigen::MatrixXf transformedVertices = mathutils::TransformPoints<Eigen::MatrixXf>((lastFrame.GetPose() * velocity).inverse(), refObject->_detection._vertices3DInRefFrame);
        std::vector<cv::Point> points2DCV = mathutils::ProjectPoints3DToPoints2D(transformedVertices, myStereoCamera);
        cv::Mat refObjectPoseMask = mathutils::Draw2DHullMaskFrom2DPointsSet(points2DCV, myStereoCamera._rows, myStereoCamera._cols);

        if (isDebug){
            cv::bitwise_or(refObjectPoseMask, displayRefObjects, displayRefObjects);
        }
        size_t countDetection = 0;
        int indexLargestOverlap = -1;
        float largestIoU = -1;
        cv::Mat currentDetectionPoseMask_Best;
        ThreeDDetection currentDetection_Best;
        for (ThreeDDetection currentDetection : currentFrame._threeDDetections){
            std::vector<cv::Point> points2DCV = mathutils::ProjectPoints3DToPoints2D(currentDetection._vertices3DInRefFrame, myStereoCamera);
            cv::Mat currentDetectionPoseMask = mathutils::Draw2DHullMaskFrom2DPointsSet(points2DCV, myStereoCamera._rows, myStereoCamera._cols);
            cv::Mat overlaps, unions;
            cv::bitwise_and(refObjectPoseMask, currentDetectionPoseMask, overlaps);
            cv::bitwise_or(refObjectPoseMask, currentDetectionPoseMask, unions);
            cv::Scalar sumOverlaps = cv::sum(overlaps);
            cv::Scalar sumUnions = cv::sum(unions);
            if ((sumOverlaps[0] / sumUnions[0]) > largestIoU && (sumOverlaps[0] / sumUnions[0]) > minIoUToReject){
                indexLargestOverlap = countDetection;
                largestIoU = (sumOverlaps[0] / sumUnions[0]);
                currentDetectionPoseMask_Best = currentDetectionPoseMask;
                currentDetection_Best.initialize(currentDetection);
            }
            // TDO_LOG_DEBUG_FORMAT("RefObject No.%d, detection No.%d, overlapping area: %d", countRefObject % countDetection % sum[0]);
            countDetection++;
            if (isDebug && countRefObject == 0){
                cv::bitwise_or(currentDetectionPoseMask, displayDetections, displayDetections);
            }
        }
        if (indexLargestOverlap >= 0){
            TDO_LOG_DEBUG("found corresponding detection for refObject " << std::to_string(countRefObject) << ", IoU :" << std::to_string(largestIoU));
            // generate new bbox for curr projections
            std::vector<std::vector<cv::Point>> currObjectContours;
            cv::findContours(currentDetectionPoseMask_Best * 255, currObjectContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            cv::Rect bboxCurrObjProj = cv::boundingRect(currObjectContours[0]);
            leftCamProjections.push_back(std::move(TwoDBoundingBox(
                bboxCurrObjProj.x + bboxCurrObjProj.width / 2,
                bboxCurrObjProj.y + bboxCurrObjProj.height / 2,
                bboxCurrObjProj.width,
                bboxCurrObjProj.height,
                refObject->_detection._pObjectInfo,
                -1.,  // Note: not used.
                std::vector<Vec2_t>()  // Note: not used.
            )));
            float disparityCurr = currentDetection_Best._pLeftBbox->_centerX - currentDetection_Best._pRightBbox->_centerX;
            rightCamProjections.push_back(std::move(TwoDBoundingBox(
                bboxCurrObjProj.x + bboxCurrObjProj.width / 2 - disparityCurr,
                bboxCurrObjProj.y + bboxCurrObjProj.height / 2,
                bboxCurrObjProj.width,  // Note: not important to be precise
                bboxCurrObjProj.height,
                refObject->_detection._pObjectInfo,
                -1.,  // Note: not used.
                std::vector<Vec2_t>()  // Note: not used.
            )));
            // generate new bbox for ref projections
            std::vector<std::vector<cv::Point>> refObjectContours;
            cv::findContours(refObjectPoseMask * 255, refObjectContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            if (refObjectContours.size() > 0){
                cv::Rect bboxRefObjProj = cv::boundingRect(refObjectContours[0]);
                leftCamProjections_ref.push_back(std::move(TwoDBoundingBox(
                    bboxRefObjProj.x + bboxRefObjProj.width / 2,
                    bboxRefObjProj.y + bboxRefObjProj.height / 2,
                    bboxRefObjProj.width,
                    bboxRefObjProj.height,
                    refObject->_detection._pObjectInfo,
                    -1.,  // Note: not used.
                    std::vector<Vec2_t>()  // Note: not used.
                )));
                float disparityRefObj = refObject->_detection._pLeftBbox->_centerX - refObject->_detection._pRightBbox->_centerX;
                rightCamProjections_ref.push_back(std::move(TwoDBoundingBox(
                    bboxRefObjProj.x + bboxRefObjProj.width / 2 - disparityRefObj,
                    bboxRefObjProj.y + bboxRefObjProj.height / 2,
                    bboxRefObjProj.width,  // Note: not important to be precise
                    bboxRefObjProj.height,
                    refObject->_detection._pObjectInfo,
                    -1.,  // Note: not used.
                    std::vector<Vec2_t>()  // Note: not used.
                )));
            }
        }
        indicesCorrespondingDetecton.push_back(indexLargestOverlap);

        countRefObject++;
    }

    if (isDebug){
        std::vector<cv::Mat> channels;
        cv::Mat zeroChannel(myStereoCamera._rows, myStereoCamera._cols, CV_8UC1, cv::Scalar(0));
        cv::Scalar sum = cv::sum(displayRefObjects);
        cv::Scalar sum2 = cv::sum(displayDetections);
        TDO_LOG_DEBUG("displayRefObjects sum: " << sum[0]);
        TDO_LOG_DEBUG("displayDetections sum: " << sum2[0]);
        channels.push_back(displayRefObjects * 255);
        channels.push_back(displayDetections * 255);
        channels.push_back(zeroChannel * 255);
        cv::Mat debugTracking;
        cv::merge(channels, debugTracking);
        std::filesystem::path debugTrackingPath = _sStereoSequencePathForDebug;
        debugTrackingPath.append("debugTracking/");
        if (!std::filesystem::exists(debugTrackingPath) && !std::filesystem::create_directory(debugTrackingPath)){
            TDO_LOG_ERROR_FORMAT("Failed to create the folder: %s", debugTrackingPath.string());
            throw std::runtime_error("Debug");
        }
        debugTrackingPath.append(mathutils::FillZeros(std::to_string(static_cast<int>(currentFrame._timestamp)), 6) + ".png");
        cv::imwrite(debugTrackingPath.string() , debugTracking);
    }

    // 3D object points in world coordinates
    std::vector<cv::Point3f> objectPoints;
    // Populate objectPoints with the corresponding 3D points from the object
    // 2D image points in image coordinates
    std::vector<cv::Point2f> imagePoints, imagePointsRef;
    // Populate imagePoints with the corresponding 2D points from the object in the image
    size_t indexRefObject = 0;
    for (int indexCorrespondingDetection : indicesCorrespondingDetecton){
        if (indexCorrespondingDetection < 0){
            indexRefObject++;
            continue;
        }
        // object center
        cv::Point3f point3D(
            refObjects[indexRefObject]->_detection._objectCenterInRefFrame(0),
            refObjects[indexRefObject]->_detection._objectCenterInRefFrame(1),
            refObjects[indexRefObject]->_detection._objectCenterInRefFrame(2)
        );
        objectPoints.push_back(point3D);
        cv::Point2f point2D(
            (*currentFrame._matchedLeftCamDetections[indexCorrespondingDetection])._centerX,
            (*currentFrame._matchedLeftCamDetections[indexCorrespondingDetection])._centerY
        );
        imagePoints.push_back(point2D);
        cv::Point2f point2DRef(
            refObjects[indexRefObject]->_detection._pLeftBbox->_centerX,
            refObjects[indexRefObject]->_detection._pLeftBbox->_centerY
        );
        imagePointsRef.push_back(point2DRef);
        indexRefObject++;
    }

    indexRefObject = 0;
    if (objectPoints.size() < 8){
        // not enough detection. Add keypts as well for tracking.
        for (int indexCorrespondingDetection : indicesCorrespondingDetecton){
            if (indexCorrespondingDetection < 0){
                indexRefObject++;
                continue;
            }
            // keypt
            cv::Point3f point3D_keypt(
                refObjects[indexRefObject]->_detection._keypt1InRefFrame(0),
                refObjects[indexRefObject]->_detection._keypt1InRefFrame(1),
                refObjects[indexRefObject]->_detection._keypt1InRefFrame(2)
            );
            objectPoints.push_back(point3D_keypt);
            cv::Point2f point2D_keypt(
                (*currentFrame._matchedLeftCamDetections[indexCorrespondingDetection])._keypts[0][0],
                (*currentFrame._matchedLeftCamDetections[indexCorrespondingDetection])._keypts[0][1]
            );
            imagePoints.push_back(point2D_keypt);

            indexRefObject++;
        }

    }

    bool isSuccess = false;
    // Estimate current frame pose using PnP
    if (objectPoints.size() < 4){
        // track failed
        velocity = Eigen::Matrix4f::Identity();
        currentFrame.SetPose(lastFrame.GetPose() * velocity);
        TDO_LOG_DEBUG("track fail. not updating camera pose.");
        // not updating cameraInWorld
    }
    else{
        // Camera intrinsic matrix (3x3)
        cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
        cameraMatrix.at<double>(0, 0) = static_cast<double>(myStereoCamera._kk(0, 0));
        cameraMatrix.at<double>(0, 2) = static_cast<double>(myStereoCamera._kk(0, 2));
        cameraMatrix.at<double>(1, 1) = static_cast<double>(myStereoCamera._kk(1, 1));
        cameraMatrix.at<double>(1, 2) = static_cast<double>(myStereoCamera._kk(1, 2));
        // Set the appropriate values for the cameraMatrix
        // Distortion coefficients
        cv::Mat distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
        // Set the appropriate values for the distCoeffs

        Mat44_t currentFrameInRefKeyFrame;
        TrackWithPnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, maxPoseError, currentFrameInRefKeyFrame);
        TDO_LOG_DEBUG("currentCameraInRefKeyFrame: \n" << currentFrameInRefKeyFrame);
        velocity = lastFrame.GetPose().inverse() * currentFrameInRefKeyFrame;  // Note: think like there is a point in last frame, first transform it to keyframe then to current frame.
        float rotAngleDeg = Eigen::AngleAxisf(velocity.block<3, 3>(0, 0)).angle() * 180.0 / M_PI;
        if (
            velocity.block(0, 3, 3, 1).norm() > maxPoseError ||
            std::abs(rotAngleDeg) > maxRotationAngleDeg
        ){
            TDO_LOG_DEBUG("track fail. not updating camera pose. velocity is \n" << velocity << ", translation is " << velocity.block(0, 3, 3, 1) << ", rotations are " << rotAngleDeg);
            // track failed
            velocity = Eigen::Matrix4f::Identity();
            currentFrame.SetPose(lastFrame.GetPose() * velocity);
            // not updating cameraInWorld
        }
        else{
            currentFrame.SetPose(currentFrameInRefKeyFrame);
            currentFrame._isTracked = true;
            currentFrame._detectionIDsOfCorrespondingRefObjects = indicesCorrespondingDetecton;
            TDO_LOG_DEBUG("track succeeded. (rotAngleDeg, " << rotAngleDeg << "). currentFrameInLast:\n" << velocity);
            isSuccess = true;
        }
    }

    if (!isSuccess && objectPoints.size() > 0){
        TDO_LOG_DEBUG("try once dense alignment");
        indexRefObject = 0;
        std::vector<TwoDBoundingBox> leftCamDetections_ref, rightCamDetections_ref, leftCamDetections, rightCamDetections;
        for (int indexCorrespondingDetection : indicesCorrespondingDetecton){
            if (indexCorrespondingDetection < 0){
                indexRefObject++;
                continue;
            }
            leftCamDetections_ref.push_back(*refObjects[indexRefObject]->_detection._pLeftBbox);
            rightCamDetections_ref.push_back(*refObjects[indexRefObject]->_detection._pRightBbox);
            leftCamDetections.push_back(*currentFrame._matchedLeftCamDetections[indexCorrespondingDetection]);
            rightCamDetections.push_back(*currentFrame._matchedRightCamDetections[indexCorrespondingDetection]);
            indexRefObject++;
        }
        cv::Mat refDepth, refImage, currDepth, currImage;
        const int imageWidth = _camera->_cols;
        const int imageHeight = _camera->_rows;
        _pPoseOptimizer->PrepareDepthAndFeatureMapFromBboxes(leftCamDetections_ref, rightCamDetections_ref, imageWidth, imageHeight, refImage, refDepth);
        _pPoseOptimizer->PrepareDepthAndFeatureMapFromBboxes(leftCamDetections, rightCamDetections, imageWidth, imageHeight, currImage, currDepth);
        if (isDebug){
            SaveCvmatForDebug(refImage, "ref");
            SaveCvmatForDebug(currImage, "curr");
        }
        Mat44_d currInRefTransform;
        _pPoseOptimizer->EstimatePose(refDepth, refImage, leftCamDetections_ref, currDepth, currImage, currInRefTransform);
        TDO_LOG_DEBUG("currentCameraInRefKeyFrame by dense align: \n" << currInRefTransform);
        velocity = lastFrame.GetPose().inverse() * currInRefTransform.cast<float>();
        float rotAngleDenseAlignDeg = Eigen::AngleAxisf(velocity.block<3, 3>(0, 0)).angle() * 180.0 / M_PI;
        if (
            velocity.block(0, 3, 3, 1).norm() > maxPoseError ||
            std::abs(rotAngleDenseAlignDeg) > maxRotationAngleDeg ||
            currInRefTransform.col(3).head<3>().norm() == 0  // Note: tracking completely failed.
        ){
            TDO_LOG_DEBUG("track by dense align fail. not updating camera pose. velocity is \n" << velocity << ", translation is " << velocity.block(0, 3, 3, 1) << ", rotations are " << rotAngleDenseAlignDeg);
            // track by dense align failed
            velocity = Eigen::Matrix4f::Identity();
            currentFrame.SetPose(lastFrame.GetPose() * velocity);
            // not updating cameraInWorld
        }
        else{
            currentFrame.SetPose(currInRefTransform.cast<float>());
            currentFrame._isTracked = true;
            currentFrame._detectionIDsOfCorrespondingRefObjects = indicesCorrespondingDetecton;
            TDO_LOG_DEBUG("track by dense align succeeded. (rotAngleDeg, " << rotAngleDenseAlignDeg << "). currentFrameInLast:\n" << velocity);
            isSuccess = true;
        }
    }

    // Note: use projected instances to do dense alignment
    if (!isSuccess && leftCamProjections.size() > 0 && currentFrame._frameID == 123){
        cv::Mat refDepth, refImage, currDepth, currImage;
        const int imageWidth = _camera->_cols;
        const int imageHeight = _camera->_rows;
        _pPoseOptimizer->PrepareDepthAndFeatureMapFromBboxes(leftCamProjections_ref, rightCamProjections_ref, imageWidth, imageHeight, refImage, refDepth);
        _pPoseOptimizer->PrepareDepthAndFeatureMapFromBboxes(leftCamProjections, rightCamProjections, imageWidth, imageHeight, currImage, currDepth);
        if (isDebug){
            SaveCvmatForDebug(refImage, "ref_2");
            SaveCvmatForDebug(currImage, "curr_2");
        }
        Mat44_d currInRefTransform;
        _pPoseOptimizer->EstimatePose(refDepth, refImage, leftCamProjections_ref, currDepth, currImage, currInRefTransform);
        velocity = currInRefTransform.cast<float>();  // Note; ref were projected to previous frame. therefore currInRef == velocity
        float rotAngleDenseAlignDeg = Eigen::AngleAxisf(velocity.block<3, 3>(0, 0)).angle() * 180.0 / M_PI;
        if (
            velocity.block(0, 3, 3, 1).norm() > maxPoseError ||
            std::abs(rotAngleDenseAlignDeg) > maxRotationAngleDeg ||
            currInRefTransform.col(3).head<3>().norm() == 0  // Note: tracking completely failed.
        ){
            TDO_LOG_DEBUG("track by dense align round2 fail. not updating camera pose. velocity is \n" << velocity << ", translation is " << velocity.block(0, 3, 3, 1) << ", rotations are " << rotAngleDenseAlignDeg);
            // track by dense align failed
            velocity = Eigen::Matrix4f::Identity();
            currentFrame.SetPose(lastFrame.GetPose() * velocity);
            // not updating cameraInWorld
        }
        else{
            Mat44_t currInKeyframeTransform = lastFrame.GetPose() * velocity;
            currentFrame.SetPose(currInKeyframeTransform);
            currentFrame._isTracked = true;
            currentFrame._detectionIDsOfCorrespondingRefObjects = indicesCorrespondingDetecton;
            TDO_LOG_DEBUG("track by dense align round2 succeeded. (rotAngleDeg, " << rotAngleDenseAlignDeg << "). currentFrameInLast:\n" << velocity);
            isSuccess = true;
        }
    }
    return isSuccess;
}

static void DrawNodes(
    cv::Mat& canvas,
    const std::vector<ThreeDDetection>& threeDDetections,
    camera::CameraBase& camera,
    const int nodeSize
){
    canvas = cv::Mat::zeros(camera._rows, camera._cols, CV_8U);
    for (ThreeDDetection oneDetection : threeDDetections){
        Eigen::MatrixXf detectionPoints3D = Eigen::MatrixXf::Zero(3, 1);
        detectionPoints3D.col(0) = oneDetection._objectCenterInRefFrame;
        std::vector<cv::Point> point2DPoseCenter = mathutils::ProjectPoints3DToPoints2D(detectionPoints3D, camera);

        cv::circle(canvas, point2DPoseCenter[0], nodeSize, cv::Scalar(255), -1, cv::LINE_AA);
    }
}

bool FrameTracker::Do2DTrackingBasedTrack(Frame& currentFrame, const Frame& lastFrame, Mat44_t& velocity, const bool isDebug) const{
    int nodeSizeHalf = 5;
    int nodeRoiSize = 20;
    int maxTrackSuccessRoiSizeError = 2;
    float maxPoseError = 0.2;

    // Draw nodes for tracking
    cv::Mat nodesCurrentFrame, nodesLastFrame;
    camera::CameraBase myStereoCamera = *_camera;
    DrawNodes(nodesCurrentFrame, currentFrame._threeDDetections, *_camera, nodeSizeHalf);
    DrawNodes(nodesLastFrame, lastFrame._threeDDetections, *_camera, nodeSizeHalf);

    cv::Mat debugTracking;
    if (isDebug){
        debugTracking = nodesCurrentFrame.clone();
    }
    std::vector<cv::Point3f> objectPoints;
    std::vector<cv::Point2f> imagePoints;
    int countRefObject = 0;
    std::vector<int> indicesCorrespondingDetecton;
    indicesCorrespondingDetecton.reserve(lastFrame._pRefKeyframe->_refObjects.size());
    currentFrame._detectionIDsOfCorrespondingRefObjects.resize(lastFrame._pRefKeyframe->_refObjects.size());
    std::fill(currentFrame._detectionIDsOfCorrespondingRefObjects.begin(), currentFrame._detectionIDsOfCorrespondingRefObjects.end(), -1);
    for (int detectionID : lastFrame._detectionIDsOfCorrespondingRefObjects){
        if (detectionID >= 0){
            ThreeDDetection theDetection = lastFrame._threeDDetections[detectionID];
            Eigen::MatrixXf detectionPoseCenter = Eigen::MatrixXf::Zero(3, 1);
            detectionPoseCenter.col(0) = theDetection._objectCenterInRefFrame;
            std::vector<cv::Point> point2DPoseCenter = mathutils::ProjectPoints3DToPoints2D(detectionPoseCenter, myStereoCamera);
            cv::Rect_<double> roiNode(point2DPoseCenter[0].x, point2DPoseCenter[0].y, nodeRoiSize, nodeRoiSize);
            cv::Ptr<cv::legacy::Tracker> twoDTracker = cv::legacy::TrackerMedianFlow::create();
            twoDTracker->init(nodesLastFrame, roiNode);
            cv::Rect_<double> roiNodeUpdate;
            twoDTracker->update(nodesCurrentFrame, roiNodeUpdate);
            TDO_LOG_DEBUG_FORMAT("roiUpdate h, w: %d, %d; nodeRoiSize: %d", roiNodeUpdate.height % roiNodeUpdate.width % nodeRoiSize);
            if (std::abs(roiNodeUpdate.width - nodeRoiSize) < maxTrackSuccessRoiSizeError && std::abs(roiNodeUpdate.height - nodeRoiSize) < maxTrackSuccessRoiSizeError){
                // track success, record all infos.
                // // object center
                cv::Point3f point3D(
                    lastFrame._pRefKeyframe->_refObjects[countRefObject]->_detection._objectCenterInRefFrame(0),
                    lastFrame._pRefKeyframe->_refObjects[countRefObject]->_detection._objectCenterInRefFrame(1),
                    lastFrame._pRefKeyframe->_refObjects[countRefObject]->_detection._objectCenterInRefFrame(2)
                );
                objectPoints.push_back(point3D);
                cv::Point2f point2D(
                    roiNodeUpdate.x,
                    roiNodeUpdate.y
                );
                imagePoints.push_back(point2D);

                if (isDebug){
                    // debug image
                    cv::rectangle(debugTracking, roiNodeUpdate, cv::Scalar(255), 2, cv::LINE_AA);
                }
                // find intercv::Mat debugTracking-frame correspondences
                float bestDistanceToDetectionCenter = std::numeric_limits<float>::infinity();
                int bestIndexDetectionCurrentFrame = -1;
                for (size_t indexDetectionCurrentFrame=0; indexDetectionCurrentFrame < currentFrame._threeDDetections.size(); indexDetectionCurrentFrame++){
                    if (
                        roiNodeUpdate.x > (currentFrame._matchedLeftCamDetections[indexDetectionCurrentFrame]->_centerX - 0.5 * currentFrame._matchedLeftCamDetections[indexDetectionCurrentFrame]->_bWidth)
                        && roiNodeUpdate.x < (currentFrame._matchedLeftCamDetections[indexDetectionCurrentFrame]->_centerX + 0.5 * currentFrame._matchedLeftCamDetections[indexDetectionCurrentFrame]->_bWidth)
                        && roiNodeUpdate.y > (currentFrame._matchedLeftCamDetections[indexDetectionCurrentFrame]->_centerY - 0.5 * currentFrame._matchedLeftCamDetections[indexDetectionCurrentFrame]->_bHeight)
                        && roiNodeUpdate.y < (currentFrame._matchedLeftCamDetections[indexDetectionCurrentFrame]->_centerY + 0.5 * currentFrame._matchedLeftCamDetections[indexDetectionCurrentFrame]->_bHeight)
                    ){
                        Eigen::Vector2f center2DRoiUpdate(roiNodeUpdate.x, roiNodeUpdate.y);
                        Eigen::Vector2f center2DCorrespondingDetectionCurrentFrame(currentFrame._matchedLeftCamDetections[indexDetectionCurrentFrame]->_centerX, currentFrame._matchedLeftCamDetections[indexDetectionCurrentFrame]->_centerY);
                        if (((center2DRoiUpdate - center2DCorrespondingDetectionCurrentFrame).norm() - bestDistanceToDetectionCenter) < bestDistanceToDetectionCenter){
                            bestIndexDetectionCurrentFrame = indexDetectionCurrentFrame;
                            bestDistanceToDetectionCenter = (center2DRoiUpdate - center2DCorrespondingDetectionCurrentFrame).norm();
                        }
                    }
                }
                if (bestIndexDetectionCurrentFrame > 0)
                    currentFrame._detectionIDsOfCorrespondingRefObjects[countRefObject] = bestIndexDetectionCurrentFrame;
            }
        }
        countRefObject++;
    }

    if(isDebug){
        std::filesystem::path debug2DTrackingPath = _sStereoSequencePathForDebug;
        debug2DTrackingPath.append("testBinaryTracking/");
        if (!std::filesystem::exists(debug2DTrackingPath) && !std::filesystem::create_directory(debug2DTrackingPath)){
            TDO_LOG_ERROR_FORMAT("Failed to create the folder: %s", debug2DTrackingPath.string());
            throw std::runtime_error(std::string("Error creating folder: ") + debug2DTrackingPath.string());
        }
        debug2DTrackingPath.append(mathutils::FillZeros(std::to_string(static_cast<int>(currentFrame._timestamp)), 6) + ".png");
        cv::imwrite(debug2DTrackingPath.string(), debugTracking);
        TDO_LOG_DEBUG_FORMAT("written testBinaryTracking debug image: %s", debug2DTrackingPath.string());
    }

    if (objectPoints.size() < 4){
        velocity = Eigen::Matrix4f::Identity();
        currentFrame.SetPose(velocity * lastFrame.GetPose());
        TDO_LOG_DEBUG("2d track fail. not updating camera pose.");
        return false;
    }
    else{
        // Camera intrinsic matrix (3x3)
        cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
        cameraMatrix.at<double>(0, 0) = static_cast<double>(myStereoCamera._kk(0, 0));
        cameraMatrix.at<double>(0, 2) = static_cast<double>(myStereoCamera._kk(0, 2));
        cameraMatrix.at<double>(1, 1) = static_cast<double>(myStereoCamera._kk(1, 1));
        cameraMatrix.at<double>(1, 2) = static_cast<double>(myStereoCamera._kk(1, 2));
        // Set the appropriate values for the cameraMatrix
        // Distortion coefficients
        cv::Mat distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
        // Set the appropriate values for the distCoeffs

        Mat44_t currentFrameInRefKeyFrame;
        TrackWithPnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, maxPoseError, currentFrameInRefKeyFrame);
        TDO_LOG_DEBUG("currentCameraInWorld: \n" << currentFrameInRefKeyFrame);
        if (
            currentFrameInRefKeyFrame.block(0, 3, 3, 1).norm() > maxPoseError
        ){
            // track failed
            velocity = Eigen::Matrix4f::Identity();
            currentFrame.SetPose(velocity * lastFrame.GetPose());
            TDO_LOG_DEBUG("track fail due to too large displacement. not updating camera pose.");
            // not updating cameraInWorld
            return false;
        }
        else{
            velocity = lastFrame.GetPose().inverse() * currentFrameInRefKeyFrame;  //Note: think like there is a point in current frame, first transform it to keyframe, then to last frame.
            currentFrame.SetPose(currentFrameInRefKeyFrame);
            currentFrame._isTracked = true;
            return true;
        }
    }

}


void FrameTracker::CreateNewLandmarks(std::shared_ptr<KeyFrame> pRefKeyFrame, std::shared_ptr<MapDataBase> pMapDb, const bool isDebug){
    float minIoUForCorrespondence = 0.6;
    float angleDiffThresholdToFit = 10;  // if landmark y axis angle diff is very small with a world prime axis (initial cam y negative axis), then fit to that prime axis. Prevent non-realistic gravity direction.
    float inPlaneDistanceThreshold = 0.07;
    // float sizeDiffToReject = 0.5;

    std::vector<std::shared_ptr<LandMark>> visibleLandmarks = pMapDb->GetVisibleLandmarks(pRefKeyFrame);  // Note: find landmarks that might fall within FoV of this keyframe.
    std::map<std::shared_ptr<LandMark>, unsigned int> observedLandmarks_indicesRefObj;
    std::vector<int> indicesLandmarkForRefObjects(pRefKeyFrame->_refObjects.size(), -1);
    std::vector<float> distancesObjectToClosestLandmark(pRefKeyFrame->_refObjects.size(), std::numeric_limits<float>::max());
    std::vector<int> indicesForClosestLandmark(pRefKeyFrame->_refObjects.size(), -1);
    size_t countNewLandmark = 0;
    cv::Mat displayVisibleLdms((*pRefKeyFrame->_pCamera)._rows, (*pRefKeyFrame->_pCamera)._cols, CV_8UC1, cv::Scalar(0));
    for (int indexRefObject=0; indexRefObject < pRefKeyFrame->_refObjects.size(); indexRefObject++){
        std::shared_ptr<RefObject> pRefObject = pRefKeyFrame->_refObjects[indexRefObject];
        std::vector<cv::Point> refObjectPoints2D = mathutils::ProjectPoints3DToPoints2D(pRefObject->_detection._vertices3DInRefFrame, (*pRefKeyFrame->_pCamera));
        cv::Mat refObjectPoseMask = mathutils::Draw2DHullMaskFrom2DPointsSet(refObjectPoints2D, (*pRefKeyFrame->_pCamera)._rows, (*pRefKeyFrame->_pCamera)._cols);
        for (int indexVisibleLandmark=0; indexVisibleLandmark < visibleLandmarks.size(); indexVisibleLandmark++){
            std::shared_ptr<LandMark> pOneLandmark = visibleLandmarks[indexVisibleLandmark];
            Eigen::MatrixXf transformedVerticesInWorld = mathutils::TransformPoints<Eigen::MatrixXf>(pOneLandmark->GetLandmarkPoseInWorld(), pOneLandmark->_vertices3DInLandmark);
            Eigen::MatrixXf transformedVerticesInCamera = mathutils::TransformPoints<Eigen::MatrixXf>((pRefKeyFrame->GetKeyframePoseInWorld()).inverse(), transformedVerticesInWorld);
            std::vector<cv::Point> oneLandmarkPoints2D = mathutils::ProjectPoints3DToPoints2D(transformedVerticesInCamera, (*pRefKeyFrame->_pCamera));
            cv::Mat oneLandmarkPoseMask = mathutils::Draw2DHullMaskFrom2DPointsSet(oneLandmarkPoints2D, (*pRefKeyFrame->_pCamera)._rows, (*pRefKeyFrame->_pCamera)._cols);
            cv::Mat overlaps, unions;
            cv::bitwise_and(refObjectPoseMask, oneLandmarkPoseMask, overlaps);
            cv::bitwise_or(refObjectPoseMask, oneLandmarkPoseMask, unions);
            cv::Scalar overlapArea = cv::sum(overlaps);
            cv::Scalar unionsArea = cv::sum(unions);
            if (isDebug && indexRefObject == 0){
                cv::bitwise_or(oneLandmarkPoseMask, displayVisibleLdms, displayVisibleLdms);
            }
            if ((overlapArea[0] / unionsArea[0]) > minIoUForCorrespondence){
                indicesLandmarkForRefObjects[indexRefObject] = indexVisibleLandmark;
                observedLandmarks_indicesRefObj[visibleLandmarks[indexVisibleLandmark]] = indexRefObject;
                break;
            }
            else if (overlapArea[0] > 0) {
                // TODO: need collision check here.
                Vec3_t objectCenterInWorld = pRefKeyFrame->GetKeyframePoseInWorld().block<3, 3>(0, 0) * pRefObject->_detection._objectCenterInRefFrame + pRefKeyFrame->GetKeyframePoseInWorld().col(3).head<3>();
                Mat44_t poseExistingLandmark = visibleLandmarks[indexVisibleLandmark]->GetLandmarkPoseInWorld();
                Eigen::Vector3f vObjectToLandmark = objectCenterInWorld - poseExistingLandmark.col(3).head<3>();
                float distanceO2L = vObjectToLandmark.norm();
                distancesObjectToClosestLandmark[indexRefObject] = distanceO2L;
                indicesForClosestLandmark[indexRefObject] = indexVisibleLandmark;
            }
            else{
                continue;
            }
        }
        float distanceThreshold;
        if (indicesForClosestLandmark[indexRefObject] >= 0){
            // found a closest landmark.
            distanceThreshold = visibleLandmarks[indicesForClosestLandmark[indexRefObject]]->_horizontalSize * 3.;  // Note: 3.0 is a factor.
        }
        else{
            // might be the initial keyframe. Or there are no visiable landmarks existing for this keyframe.
            distanceThreshold = 0;
        }
        if (indicesLandmarkForRefObjects[indexRefObject] < 0){
            if (distancesObjectToClosestLandmark[indexRefObject] > distanceThreshold){  // if within certain physical distance, still create correspondence.
                // if not correspondence, create landmark.
                Mat44_t poseLandmarkInWorld;
                Vec3_t keypt1InLandmark;
                LandMark::ComputeLandmarkPoseInWorldAndKeypt1InWolrd(pRefKeyFrame, pRefObject, poseLandmarkInWorld, keypt1InLandmark);

                // do not create gravity non-realistic landmark
                Vec3_t keypt1InWorld = poseLandmarkInWorld.block<3, 3>(0, 0) * keypt1InLandmark + poseLandmarkInWorld.col(3).head<3>();
                Vec3_t vecPt1InWorld = keypt1InWorld - poseLandmarkInWorld.col(3).head<3>();
                vecPt1InWorld = vecPt1InWorld / vecPt1InWorld.norm();
                float angleDiffWithWorldAxis_deg = std::acos(-vecPt1InWorld(1)) / M_PI * 180.0;
                if (angleDiffWithWorldAxis_deg > angleDiffThresholdToFit){
                    continue;
                }

                // // if too far from ground plane, skip
                // std::vector<std::shared_ptr<LandMark>> allLandmarksInDb = pMapDb->GetAllLandmarks();
                // if (allLandmarksInDb.size() >= 5){
                //     std::vector<Vec3_t> points(allLandmarksInDb.size());
                //     for (auto pLandmark : allLandmarksInDb) {
                //         points.push_back(pLandmark->GetLandmarkPoseInWorld().col(3).head<3>());
                //     }
                //     pcl::ModelCoefficients::Ptr planeCoeff(new pcl::ModelCoefficients);
                //     pcl::PointIndices::Ptr pIndicesInliers(new pcl::PointIndices);
                //     mathutils::EstimatePlaneFromPoints(points, inPlaneDistanceThreshold, planeCoeff, pIndicesInliers);
                //     if (pIndicesInliers->indices.size() > (float)allLandmarksInDb.size() * 0.8){
                //         pcl::PointXYZ curr = pcl::PointXYZ(poseLandmarkInWorld(0, 3), poseLandmarkInWorld(1, 3), poseLandmarkInWorld(2, 3));
                //         float distanceCurrToGround = mathutils::ComputeDistanceFromPlane(planeCoeff, curr);
                //         if (distanceCurrToGround > inPlaneDistanceThreshold){
                //             TDO_LOG_DEBUG("skip landmark due to distanceCurrToGround: " << distanceCurrToGround);
                //             continue;
                //         }
                //     }
                // }

                // // if size too different, abandon
                // Vec3_t objectSize = pMapDb->GetObjectSize();
                // if (pRefObject->_detection._horizontalSize < objectSize.minCoeff() * (1 - sizeDiffToReject)){
                //     continue;
                // }
                // else if (pRefObject->_detection._horizontalSize > objectSize.minCoeff() * (1 + sizeDiffToReject)){
                //     continue;
                // }
                // else if ((keypt1InLandmark.norm() * 2) < objectSize.maxCoeff() * (1 - sizeDiffToReject)){
                //     continue;
                // }
                // else if ((keypt1InLandmark.norm() * 2) > objectSize.maxCoeff() * (1 + sizeDiffToReject)){
                //     continue;
                // }

                std::shared_ptr<object::ObjectBase> pObjectInfo = pRefObject->_detection._pObjectInfo;
                std::shared_ptr<LandMark> pOneLandmark = std::make_shared<LandMark>(
                    poseLandmarkInWorld,
                    keypt1InLandmark,
                    pRefObject->_detection._horizontalSize,
                    pObjectInfo
                );
                pOneLandmark->AddObservation(pRefKeyFrame, indexRefObject);
                pMapDb->AddLandMark(pOneLandmark);
                observedLandmarks_indicesRefObj[pOneLandmark] = indexRefObject;
                TDO_LOG_DEBUG_FORMAT("Failed matching correspondence (distance %f). Creating new landmark...", distancesObjectToClosestLandmark[indexRefObject]);
                countNewLandmark++;
                pRefKeyFrame->_bContainNewLandmarks = true;
                continue;
            }
            indicesLandmarkForRefObjects[indexRefObject] = indicesForClosestLandmark[indexRefObject];
            TDO_LOG_DEBUG_FORMAT("Resurrect correspondence due to close 3d distance (%f m).", distancesObjectToClosestLandmark[indexRefObject]);
        }
        // if correspondence, and new keyframe is closer, update landmark pose.
        // TODO: should use multi-view stereo to update landmark pose. need stereo rectification and check if object is within FoV after stereo recti.
        visibleLandmarks[indicesLandmarkForRefObjects[indexRefObject]]->AddObservation(pRefKeyFrame, indexRefObject);
        observedLandmarks_indicesRefObj[visibleLandmarks[indicesLandmarkForRefObjects[indexRefObject]]] = indexRefObject;
    }

    if (isDebug){
        std::filesystem::path debugLdmProj = _sStereoSequencePathForDebug;
        debugLdmProj.append("debugLdmProj/");
        if (!std::filesystem::exists(debugLdmProj) && !std::filesystem::create_directory(debugLdmProj)){
            TDO_LOG_ERROR_FORMAT("Failed to create the folder: %s", debugLdmProj.string());
        }
        else{
            debugLdmProj.append(mathutils::FillZeros(std::to_string(static_cast<int>(pRefKeyFrame->_keyFrameID)), 6) + ".png");

            cv::imwrite(debugLdmProj.string() , displayVisibleLdms * 255);
        }
    }
    pRefKeyFrame->InitializeObservedLandmarks(observedLandmarks_indicesRefObj);
    TDO_LOG_INFO_FORMAT("Created %d new landmarks in keyframe %d. \nTotally %d landmarks currently in MapDb.", countNewLandmark % pRefKeyFrame->_keyFrameID % pMapDb->_landmarks.size());

}

}  // end of namespace eventobjectslam
