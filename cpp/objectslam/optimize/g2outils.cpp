#include "g2outils.h"

#include <logging.h>
TDO_LOGGER("objectslam.optimize.g2outils")

namespace eventobjectslam {

namespace optimize {

namespace g2outils {

static ::g2o::SE3Quat ConvertToG2oSE3(const Mat44_d& campose) {
    Mat33_d rot;
    rot << campose(0, 0), campose(0, 1), campose(0, 2),
           campose(1, 0), campose(1, 1), campose(1, 2),
           campose(2, 0), campose(2, 1), campose(2, 2);
    Vec3_d trans;
    trans << campose(0, 3), campose(1, 3), campose(2, 3);
    return ::g2o::SE3Quat{rot, trans};
}

ShotVertex* CreateShotVertex(const unsigned int vtxId, const Mat44_d& worldToShotTransform, const bool isConstant) {
    // create vertex
    auto vtx = new ShotVertex();
    vtx->setId(vtxId);
    vtx->setEstimate(ConvertToG2oSE3(worldToShotTransform));
    vtx->setFixed(isConstant);
    return vtx;
}

ShotVertexSE2* CreateShotVertexSE2(const unsigned int vtxId, const Mat44_d& worldToShotTransform, const bool isConstant){
    ShotVertexSE2* vtx = new ShotVertexSE2();
    vtx->setId(vtxId);
    double Ry = mathutils::GetRyFromPose<double>(worldToShotTransform);
    ::g2o::SE2 initialEstimate(worldToShotTransform(0, 3), worldToShotTransform(2, 3), Ry);  // Note: (tx, tz, Ry)
    vtx->setEstimate(initialEstimate);
    vtx->setFixed(isConstant);
    return vtx;
}

LandmarkPointVertex* CreateLandmarkPointVertex(const unsigned int vtxId, const Vec3_d& posInWorld, const bool isConstant) {
    // vertexを作成
    auto vtx = new LandmarkPointVertex();
    vtx->setId(vtxId);
    vtx->setEstimate(posInWorld);
    vtx->setFixed(isConstant);
    vtx->setMarginalized(true);
    return vtx;
}

LandmarkPointVertex2D* CreateLandmarkPointVertex2D(const unsigned int vtxId, const Vec3_d& posInWorld, const bool isConstant) {
    LandmarkPointVertex2D* vtx = new LandmarkPointVertex2D();
    vtx->setId(vtxId);
    Vec2_d pos_xz(posInWorld(0), posInWorld(2));
    vtx->setEstimate(pos_xz);
    vtx->setFixed(isConstant);
    vtx->setMarginalized(true);
    return vtx;
}

/** When do BA in SE2, need to reconstruct SE3 camera pose from SE2 vertex estimate.
 */
const Mat44_t ReconstructNewCameraPoseInWorld(ShotVertexSE2* pShotVtx, const Mat44_t& oldCameraPoseInWorld) {
    const ::g2o::SE2& se2_cw = pShotVtx->estimate();
    Mat44_t newcamera_pose_cw = Mat44_t::Identity();
    Mat44_t oldcamera_pose_cw = oldCameraPoseInWorld.inverse();
    newcamera_pose_cw.block<3, 1>(0, 3) = oldcamera_pose_cw.block<3, 1>(0, 3);
    Eigen::AngleAxis<float> aaRy(se2_cw.rotation().angle(), Eigen::Matrix<float, 3, 1>::UnitY());
    newcamera_pose_cw.block<3, 3>(0, 0) = aaRy.toRotationMatrix();
    newcamera_pose_cw(0, 3) = (float)se2_cw.translation().x();
    newcamera_pose_cw(2, 3) = (float)se2_cw.translation().y();
    return newcamera_pose_cw.inverse();
}

/** When do BA in SE2, need to reconstruct vec3 landmark point from vec2 landmark vertex estimate.
 */
const Vec3_t ReconstructNewPointInWorld(LandmarkPointVertex2D* pLmVtx, const Vec3_t& oldLmPointInWorld) {
    const Vec2_d& pos_xz = pLmVtx->estimate();
    Vec3_t newPointInWorld = oldLmPointInWorld;
    newPointInWorld(0) = (float)pos_xz(0);
    newPointInWorld(2) = (float)pos_xz(1);
    return newPointInWorld;
}

void StereoPerspectiveReprojEdge::linearizeOplus() {
    auto vj = static_cast<ShotVertex*>(_vertices.at(1));
    const ::g2o::SE3Quat cam_pose_cw = vj->ShotVertex::estimate();

    auto vi = static_cast<LandmarkPointVertex*>(_vertices.at(0));
    const Vec3_d pos_w = vi->LandmarkPointVertex::estimate();
    const Vec3_d pos_c = cam_pose_cw.map(pos_w);

    const auto x = pos_c(0);
    const auto y = pos_c(1);
    const auto z = pos_c(2);
    const auto z_sq = z * z;

    const Mat33_d rot_cw = cam_pose_cw.rotation().toRotationMatrix();

    _jacobianOplusXi(0, 0) = -fx_ * rot_cw(0, 0) / z + fx_ * x * rot_cw(2, 0) / z_sq;
    _jacobianOplusXi(0, 1) = -fx_ * rot_cw(0, 1) / z + fx_ * x * rot_cw(2, 1) / z_sq;
    _jacobianOplusXi(0, 2) = -fx_ * rot_cw(0, 2) / z + fx_ * x * rot_cw(2, 2) / z_sq;

    _jacobianOplusXi(1, 0) = -fy_ * rot_cw(1, 0) / z + fy_ * y * rot_cw(2, 0) / z_sq;
    _jacobianOplusXi(1, 1) = -fy_ * rot_cw(1, 1) / z + fy_ * y * rot_cw(2, 1) / z_sq;
    _jacobianOplusXi(1, 2) = -fy_ * rot_cw(1, 2) / z + fy_ * y * rot_cw(2, 2) / z_sq;

    _jacobianOplusXi(2, 0) = _jacobianOplusXi(0, 0) - focal_x_baseline_ * rot_cw(2, 0) / z_sq;
    _jacobianOplusXi(2, 1) = _jacobianOplusXi(0, 1) - focal_x_baseline_ * rot_cw(2, 1) / z_sq;
    _jacobianOplusXi(2, 2) = _jacobianOplusXi(0, 2) - focal_x_baseline_ * rot_cw(2, 2) / z_sq;

    _jacobianOplusXj(0, 0) = x * y / z_sq * fx_;
    _jacobianOplusXj(0, 1) = -(1.0 + (x * x / z_sq)) * fx_;
    _jacobianOplusXj(0, 2) = y / z * fx_;
    _jacobianOplusXj(0, 3) = -1.0 / z * fx_;
    _jacobianOplusXj(0, 4) = 0.0;
    _jacobianOplusXj(0, 5) = x / z_sq * fx_;

    _jacobianOplusXj(1, 0) = (1.0 + y * y / z_sq) * fy_;
    _jacobianOplusXj(1, 1) = -x * y / z_sq * fy_;
    _jacobianOplusXj(1, 2) = -x / z * fy_;
    _jacobianOplusXj(1, 3) = 0.0;
    _jacobianOplusXj(1, 4) = -1.0 / z * fy_;
    _jacobianOplusXj(1, 5) = y / z_sq * fy_;

    _jacobianOplusXj(2, 0) = _jacobianOplusXj(0, 0) - focal_x_baseline_ * y / z_sq;
    _jacobianOplusXj(2, 1) = _jacobianOplusXj(0, 1) + focal_x_baseline_ * x / z_sq;
    _jacobianOplusXj(2, 2) = _jacobianOplusXj(0, 2);
    _jacobianOplusXj(2, 3) = _jacobianOplusXj(0, 3);
    _jacobianOplusXj(2, 4) = 0;
    _jacobianOplusXj(2, 5) = _jacobianOplusXj(0, 5) - focal_x_baseline_ / z_sq;

}

void StereoPerspectiveReprojEdgeSE2::linearizeOplus() {
    const ShotVertexSE2* vPose = static_cast<const ShotVertexSE2*>(_vertices.at(1));
    const ::g2o::SE2 cam_pose_xzRy = vPose->ShotVertexSE2::estimate();
    Vec2_d t = cam_pose_xzRy.translation();  // t = (1.0, 2.0)
    double Ry = cam_pose_xzRy.rotation().angle();
    Eigen::Matrix3d R = Eigen::AngleAxisd(Ry, Eigen::Vector3d::UnitY()).toRotationMatrix();
    Mat44_d m_cam_pose_cw = Mat44_d::Identity();
    m_cam_pose_cw.block<3, 3>(0, 0) = R;
    m_cam_pose_cw(0, 3) = t(0);
    m_cam_pose_cw(1, 3) = -1 * _Y_ws;  // -1 is for wc->cw, while Y axes are parallel for c and w frames.
    m_cam_pose_cw(2, 3) = t(1);
    const ::g2o::SE3Quat cam_pose_cw = ConvertToG2oSE3(m_cam_pose_cw);

    const LandmarkPointVertex2D* vPoint = static_cast<LandmarkPointVertex2D*>(_vertices.at(0));
    const Vec2_d pos_wp_xz = vPoint->LandmarkPointVertex2D::estimate();
    const Vec3_d pos_wp_xyz(pos_wp_xz(0), _Y_wp, pos_wp_xz(1));
    const Vec3_d pos_c = cam_pose_cw.map(pos_wp_xyz);

    const auto x = pos_c(0);
    const auto y = pos_c(1);
    const auto z = pos_c(2);
    const auto z_sq = z * z;

    const Mat33_d rot_cw = cam_pose_cw.rotation().toRotationMatrix();

    double c, s, deltaX, deltaZ, Xc, Zc, Z2, Y0, b;
    c = std::cos(Ry);
    s = std::sin(Ry);
    deltaX = pos_wp_xyz(0) - t(0);
    deltaZ - pos_wp_xyz(2) - t(1);
    Xc = c * deltaX - s * deltaZ;
    Zc = s * deltaX + c * deltaZ;
    Z2 = Zc * Zc;
    Y0 = _Y_wp - _Y_ws;
    b = (focal_x_baseline_ / fx_);

    // ∂uL / ∂Xw , ∂uL / ∂Zw
    _jacobianOplusXi(0,0) = fx_ * (  c*Zc - s*Xc) / Z2;
    _jacobianOplusXi(0,1) = fx_ * ( -s*Zc - c*Xc) / Z2;
    // ∂vL / ∂Xw , ∂vL / ∂Zw
    _jacobianOplusXi(1,0) = -fy_ * Y0 * s / Z2;
    _jacobianOplusXi(1,1) = -fy_ * Y0 * c / Z2;
    // ∂uR / ∂Xw , ∂uR / ∂Zw
    _jacobianOplusXi(2,0) = fx_ * (c*Zc - s*(Xc - b)) / Z2;
    _jacobianOplusXi(2,1) = fx_ * (-s*Zc - c*(Xc - b)) / Z2;

    // --- uL ---
    _jacobianOplusXj(0,0) = fx_ * (-c*Zc + s*Xc) / Z2;
    _jacobianOplusXj(0,1) = fx_ * (s*Zc + c*Xc) / Z2;
    _jacobianOplusXj(0,2) = fx_ * ((-s*deltaX - c*deltaZ)*Zc - Xc*( c*deltaX - s*deltaZ)) / Z2;
    // --- vL ---
    _jacobianOplusXj(1,0) = fy_ * Y0 * s / Z2;
    _jacobianOplusXj(1,1) = fy_ * Y0 * c / Z2;
    _jacobianOplusXj(1,2) = -fy_ * Y0 * (c*deltaZ - s*deltaZ) / Z2;
    // --- uR (replace Xc → Xc-b) ---
    const double XcR = Xc - b;
    _jacobianOplusXj(2,0) = fx_ * (-c*Zc + s*XcR) / Z2;
    _jacobianOplusXj(2,1) = fx_ * ( s*Zc + c*XcR) / Z2;
    _jacobianOplusXj(2,2) = fx_ * ((-s*deltaX - c*deltaZ)*Zc - XcR*( c*deltaX - s*deltaZ)) / Z2;

}



// ====================== read / write virtual methods overwrite =====================

bool ShotVertex::read(std::istream& is) {
    Vec7_d estimate;
    for (unsigned int i = 0; i < 7; ++i) {
        is >> estimate(i);
    }
    ::g2o::SE3Quat g2o_cam_pose_wc;
    g2o_cam_pose_wc.fromVector(estimate);
    setEstimate(g2o_cam_pose_wc.inverse());
    return true;
}

bool LandmarkPointVertex::read(std::istream& is) {
    Vec3_d lv;
    for (unsigned int i = 0; i < 3; ++i) {
        is >> _estimate(i);
    }
    return true;
}

bool LandmarkPointVertex::write(std::ostream& os) const {
    const Vec3_d pos_w = estimate();
    for (unsigned int i = 0; i < 3; ++i) {
        os << pos_w(i) << " ";
    }
    return os.good();
}

bool ShotVertex::write(std::ostream& os) const {
    ::g2o::SE3Quat g2o_cam_pose_wc(estimate().inverse());
    for (unsigned int i = 0; i < 7; ++i) {
        os << g2o_cam_pose_wc[i] << " ";
    }
    return os.good();
}

bool StereoPerspectiveReprojEdge::read(std::istream& is) {
    for (unsigned int i = 0; i < 3; ++i) {
        is >> _measurement(i);
    }
    for (unsigned int i = 0; i < 3; ++i) {
        for (unsigned int j = i; j < 3; ++j) {
            is >> information()(i, j);
            if (i != j) {
                information()(j, i) = information()(i, j);
            }
        }
    }
    return true;
}

bool StereoPerspectiveReprojEdge::write(std::ostream& os) const {
    for (unsigned int i = 0; i < 3; ++i) {
        os << measurement()(i) << " ";
    }
    for (unsigned int i = 0; i < 3; ++i) {
        for (unsigned int j = i; j < 3; ++j) {
            os << " " << information()(i, j);
        }
    }
    return os.good();
}

}  // end of namespace g2outils

}  // end of namespace optimize

}  // end of namespace eventobejctslam