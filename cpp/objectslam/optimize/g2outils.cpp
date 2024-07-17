#include "g2outils.h"

#include <logging.h>
TDO_LOGGER("objectslam.optimize.g2outils")

namespace eventobjectslam {

namespace optimize {

namespace g2outils {

static ::g2o::SE3Quat ConvertToG2oSE3(const Mat44_d& campose) {
    const Mat33_d rot = campose.block<3, 3>(0, 0);
    const Vec3_d trans = campose.block<3, 1>(0, 3);
    return ::g2o::SE3Quat(rot, trans);
}

ShotVertex* CreateShotVertex(const unsigned int vtxId, const Mat44_d& worldToShotTransform, const bool isConstant) {
    // create vertex
    auto vtx = new ShotVertex();
    vtx->setId(vtxId);
    ::g2o::SE3Quat qPose_cw = ConvertToG2oSE3(worldToShotTransform);
    vtx->setEstimate(qPose_cw);
    vtx->setFixed(isConstant);
    return vtx;
}

VertexLandmarkCylinder* CreateLandmarkCylinderVertex(const unsigned int vtxId, const Mat44_d& poseInWorld, const Vec3_d& cylinderHalfSizes, const Vec3_d& keypt1InLocal, const bool isConstant){
    CylinderTarget cylinder;
    cylinder.initialize(cylinderHalfSizes, keypt1InLocal);
    cylinder.poseInWorld = ConvertToG2oSE3(poseInWorld);
    auto vtx = new VertexLandmarkCylinder();
    vtx->initialize(cylinder);
    vtx->setId(vtxId);
    vtx->setEstimate(cylinder);
    vtx->setFixed(isConstant);
    vtx->setMarginalized(true);
    return vtx;
}


// analytical jacobian faster than auto differentiation in g2o.
void StereoPerspectiveReprojEdge::linearizeOplus() {
    auto vj = static_cast<ShotVertex*>(_vertices.at(1));
    const ::g2o::SE3Quat& cam_pose_cw = vj->ShotVertex::estimate();

    auto vi = static_cast<VertexLandmarkCylinder*>(_vertices.at(0));
    const Vec3_d& pos_w = vi->VertexLandmarkCylinder::estimate().poseInWorld.translation();
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

/*--------------------------------------------------------------------------*/
/*                  5D cylinder based optimization                          */
/*--------------------------------------------------------------------------*/
void VertexLandmarkCylinder::oplusImpl(const double *update_)
{
    Eigen::Map<const Vec6_d> update(update_);
    CylinderTarget cylinder_b = estimate();
    Vec6_d update_noRz = update;
    update_noRz(2) *= 0; // Note: rotation around z is ignored.
    cylinder_b.poseInWorld = ::g2o::SE3Quat::exp(update_noRz) * cylinder_b.poseInWorld;
    setEstimate(cylinder_b);
}

void EdgeSE3CylinderProj::computeError()
{
    const ShotVertex *shotVertex = dynamic_cast<const ShotVertex*>(_vertices[1]);  // Note: pose world to cam
    const VertexLandmarkCylinder *cylinderVertex = dynamic_cast<const VertexLandmarkCylinder*>(_vertices[0]);  // Note: pose object to world.
    ::g2o::SE3Quat cam_pose_Tcw = shotVertex->estimate();
    CylinderTarget cylinder_world = cylinderVertex->estimate();    

    Vec4_d keypts = cylinder_world.ProjectToImageKeypts(cam_pose_Tcw, kk);
    // right cam
    Vec3_d cylinder_pos_cam = cam_pose_Tcw.map(cylinder_world.poseInWorld.translation());
    double reproj_x_right = keypts(2) - focal_x_baseline_ / cylinder_pos_cam(2);

    Vec5_d keyptsAndRightX;
    keyptsAndRightX.head<4>() = keypts;
    keyptsAndRightX(4) = reproj_x_right;

    _error = keyptsAndRightX - _measurement;
}

double EdgeSE3CylinderProj::get_error_norm()
{
    computeError();
    return _error.norm();
}


}  // end of namespace g2outils

}  // end of namespace optimize

}  // end of namespace eventobejctslam