#ifndef EVENTOBJECTSLAM_TRAJECTORYSMOOTHER_H
#define EVENTOBJECTSLAM_TRAJECTORYSMOOTHER_H

#include "objectslam.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace eventobjectslam {

// Online port of tool_evalAndViszData/smooth_traj.py: a velocity-adaptive constant-velocity Kalman filter
// with a burn-in bypass phase and innovation gating. Translation is smoothed directly; rotation is smoothed
// in rotation-vector space (with quaternion sign continuity) and converted back.
// Note: the offline script's burn-out phase requires knowing the trajectory end and is omitted online.
class ConstantVelocityKalman3D {
public:
    // Note: DontAlign on stored matrices avoids imposing 32-byte over-alignment on every object
    // holding this filter (the containing SLAMSystem lives on the stack).
    using Vec6d = Eigen::Matrix<double, 6, 1, Eigen::DontAlign>;
    using Mat66d = Eigen::Matrix<double, 6, 6, Eigen::DontAlign>;

    // isMeasurementValid=false (e.g. tracking failed) advances the prediction without a measurement update.
    Eigen::Vector3d Filter(const Eigen::Vector3d& measurement, const bool isMeasurementValid){
        if (_numMeasurements == 0){
            _x.setZero();
            _x.head<3>() = measurement;
            _P.setZero();
            _P.diagonal() << 0.01, 0.01, 0.01, 0.1, 0.1, 0.1;
            _vNormSmoothed = 1e-3;
            _numMeasurements++;
            return measurement;
        }
        if (_numMeasurements == 1){
            // clean two-frame initialization.
            _x.tail<3>() = measurement - _x.head<3>();
            _P.setZero();
            _P.diagonal() << 0.01, 0.01, 0.01, 0.05, 0.05, 0.05;
            _vNormSmoothed = std::max(_x.tail<3>().norm(), 1e-3);
        }
        const size_t frameIndex = _numMeasurements;
        _numMeasurements++;

        // covariance mode: burn-in bypass vs velocity-adaptive.
        double q_p, q_v, r_m;
        if (frameIndex < _burnInFrames){
            q_p = 1e10;  // 1e5^2
            q_v = 1e10;
            r_m = 0.0;
        }
        else{
            const double vNormRaw = _x.tail<3>().norm();
            _vNormSmoothed = std::max(0.15 * vNormRaw + 0.85 * _vNormSmoothed, 1e-3);
            q_p = std::pow(_sigmaP * _vNormSmoothed + 0.005, 2);
            q_v = std::pow(_sigmaV * _vNormSmoothed + 0.0005, 2);
            r_m = std::pow(_sigmaM * _vNormSmoothed + 0.01, 2);
        }

        // predict (constant velocity, dt = 1 frame).
        Mat66d F = Mat66d::Identity();
        F.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
        _x = F * _x;
        _P = (F * _P * F.transpose()).eval();
        _P.diagonal().head<3>().array() += q_p;
        _P.diagonal().tail<3>().array() += q_v;

        // innovation and gating (spikes keep the smooth prediction).
        const Eigen::Vector3d innovation = measurement - _x.head<3>();
        const bool isGatedOut = frameIndex >= _burnInFrames && innovation.norm() > _gateThreshold;
        if (isMeasurementValid && !isGatedOut){
            Eigen::Matrix3d S = _P.topLeftCorner<3, 3>();
            S.diagonal().array() += r_m;
            const Eigen::Matrix<double, 6, 3, Eigen::DontAlign> K = _P.leftCols<3>() * S.inverse();
            _x += K * innovation;
            Mat66d I_KH = Mat66d::Identity();
            I_KH.leftCols<3>() -= K;  // H = [I3 0]
            _P = (I_KH * _P).eval();
        }
        return _x.head<3>();
    }

private:
    static constexpr double _sigmaP = 0.02;
    // Measurement-noise scale: the main smoothness knob (higher = smoother but more lag behind fast
    // motion). 5.0 halves the exported-trajectory roughness (RMS frame-to-frame jerk) vs the original
    // 0.1, measured on the real unitree seq12_retrain sequence: 0.0153 -> 0.0078, with per-frame
    // corrections (lag proxy) of 1-3 cm.
    static constexpr double _sigmaM = 5.0;
    static constexpr double _sigmaV = 0.002;
    static constexpr size_t _burnInFrames = 15;
    static constexpr double _gateThreshold = 0.3;

    Vec6d _x = Vec6d::Zero();
    Mat66d _P = Mat66d::Zero();
    double _vNormSmoothed = 1e-3;
    size_t _numMeasurements = 0;
};

class TrajectorySmoother {
public:
    Mat44_t Smooth(const Mat44_t& poseInWorld, const bool isMeasurementValid){
        const Eigen::Vector3d position = poseInWorld.col(3).head<3>().cast<double>();
        Eigen::Quaterniond quaternion(poseInWorld.block<3, 3>(0, 0).cast<double>());
        // sign continuity: q and -q encode the same rotation; avoids rotation-vector discontinuities.
        if (_hasLastQuaternion && _lastQuaternion.dot(quaternion) < 0){
            quaternion.coeffs() *= -1;
        }
        _lastQuaternion = quaternion;
        _hasLastQuaternion = true;
        const Eigen::AngleAxisd angleAxis(quaternion);
        const Eigen::Vector3d rotationVector = angleAxis.angle() * angleAxis.axis();

        const Eigen::Vector3d smoothedPosition = _positionFilter.Filter(position, isMeasurementValid);
        const Eigen::Vector3d smoothedRotationVector = _rotationFilter.Filter(rotationVector, isMeasurementValid);

        Mat44_t smoothedPose = Mat44_t::Identity();
        const double smoothedAngle = smoothedRotationVector.norm();
        if (smoothedAngle > 1e-12){
            smoothedPose.block<3, 3>(0, 0) = Eigen::AngleAxisd(smoothedAngle, smoothedRotationVector / smoothedAngle).toRotationMatrix().cast<float>();
        }
        smoothedPose.col(3).head<3>() = smoothedPosition.cast<float>();
        return smoothedPose;
    }

private:
    ConstantVelocityKalman3D _positionFilter;
    ConstantVelocityKalman3D _rotationFilter;
    // Note: DontAlign — a plain Quaterniond member (32 bytes) would impose over-alignment on containers.
    Eigen::Quaternion<double, Eigen::DontAlign> _lastQuaternion = Eigen::Quaterniond::Identity();
    bool _hasLastQuaternion = false;
};

}  // end of namespace eventobjectslam

#endif  // EVENTOBJECTSLAM_TRAJECTORYSMOOTHER_H
