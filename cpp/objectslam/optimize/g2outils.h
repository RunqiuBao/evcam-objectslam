#ifndef EVENTOBJECTSLAM_OPTIMIZE_G2OUTILS_H
#define EVENTOBJECTSLAM_OPTIMIZE_G2OUTILS_H

#include "objectslam.h"
#include "landmark.h"
#include "keyframe.h"
#include "mathutils.h"

#include <g2o/core/solver.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/core/optimization_algorithm_levenberg.h>

#include <g2o/core/base_multi_edge.h>

namespace eventobjectslam {

namespace optimize {

namespace g2outils {

class ShotVertex final : public ::g2o::BaseVertex<6, ::g2o::SE3Quat> {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ShotVertex()
    : BaseVertex<6, ::g2o::SE3Quat>() {}

    bool read(std::istream& is) override;

    bool write(std::ostream& os) const override;

    void setToOriginImpl() override {
        _estimate = ::g2o::SE3Quat();
    }

    void oplusImpl(const ::g2o::number_t* update_) override {
        Eigen::Map<const Vec6_d> update(update_);
        setEstimate(::g2o::SE3Quat::exp(update) * estimate());
    }

};

/*--------------------------------------------------------------------------*/
/*                  5D cylinder based optimization                          */
/*--------------------------------------------------------------------------*/
class CylinderTarget
{
public:
    ::g2o::SE3Quat poseInWorld;  // 6 dof object. But Rz will be degenerated.
    Vec3_d halfSizes;  // x, y size are same.
    Vec3_d keypt1InLocal;

    CylinderTarget(){
        poseInWorld = ::g2o::SE3Quat();
        halfSizes.setZero();
        keypt1InLocal.setZero();
    }

    CylinderTarget& operator=(const CylinderTarget& cylinder) {
        poseInWorld = cylinder.poseInWorld;
        halfSizes = cylinder.halfSizes;
        keypt1InLocal = cylinder.keypt1InLocal;
        return *this;
    }

    void initialize(const Vec3_d& cylinderHalfSizes, const Vec3_d& keypt1){
        halfSizes = cylinderHalfSizes;
        keypt1InLocal = keypt1;
    }

    // // x,y,z,roll,pitch,yaw
    // inline void fromMinimalVector(const Vec6_d &v){
    //     Eigen::Quaterniond posequat = mathutils::zyx_euler_to_quat<double>(v(3), v(4), v(5));
    //     this->poseInWorld = ::g2o::SE3Quat(posequat, v.head<3>());
    // }

	// inline const Vec3_d &translation() const { return poseInWorld.translation(); }
	// inline void setTranslation(const Vec3_d &t_) {
        // poseInWorld.setTranslation(t_); 
    // }
	// inline void setRotation(const Eigen::Quaterniond &r_) {
        // poseInWorld.setRotation(r_);
    // }
	// inline void setRotation(const Mat33_d &R) {
        // poseInWorld.setRotation(Eigen::Quaterniond(R));
    // }

    // // update current cylinder pose with Lie algebra exponential map.
    // CylinderTarget exp_update(const Vec6_d& update){  // Note: update = omega (rotate) + upsilon (trans).
        // CylinderTarget newCylinder;
        // newCylinder.initialize(this->halfSizes, this->keypt1InLocal);
        // newCylinder.poseInWorld = this->poseInWorld * ::g2o::SE3Quat::exp(update);
        // return newCylinder;
    // }

    // // compare error between two cylinders
    // Vec6_d cylinder_log_error(const CylinderTarget& cylinder_b) const {
        // Vec6_d res;
        // ::g2o::SE3Quat pose_diff = cylinder_b.poseInWorld.inverse() * this->poseInWorld;
        // res = pose_diff.log();
        // return res;
    // }

    // // api func required by g2o
    // Vec6_d min_log_error(const CylinderTarget& cylinder_b, bool print_details = false) const {
        // return cylinder_log_error(cylinder_b);
    // }
 
    // inline Vec6_d toMinimalVector() const {
        // Vec6_d v;
        // v = mathutils::ConvertToXYZPRYVector<::g2o::SE3Quat>(this->poseInWorld);
        // return v;
    // }

    Mat44_d GetSimilarityTransform() const {
        Mat44_d transform = this->poseInWorld.to_homogeneous_matrix();
        Mat33_d scale_mat = this->halfSizes.asDiagonal();
        transform.topLeftCorner<3, 3>() = transform.topLeftCorner<3, 3>() * scale_mat;
        return transform;
    }

    // use 8 vertices to represent a cylinder.
    Mat3Xd compute_3D_cylinder_vertices() const {
        Mat3Xd vertices_body;
        vertices_body.resize(3, 8);
        vertices_body << 1, 1, -1, -1, 1, 1, -1, -1,
                         1, -1, -1, 1, 1, -1, -1, 1,
                         -1, -1, -1, -1, 1, 1, 1, 1;
        Mat3Xd corners_world = mathutils::homo_to_real_coord<double>(GetSimilarityTransform() * mathutils::real_to_homo_coord<double>(vertices_body));
        return corners_world;
    }

    Mat3Xd compute_keypt_3D() const {
        Mat3Xd keypts_local;
        keypts_local.resize(3, 1);
        keypts_local << 0,
                        0,
                        1;
        Mat3Xd keypts_world = mathutils::homo_to_real_coord<double>(GetSimilarityTransform() * mathutils::real_to_homo_coord<double>(keypts_local));
        return keypts_world;
    }

    Mat3Xd compute_keypts_3D() const {
        Mat3Xd keypts_local;
        keypts_local.resize(3, 2);
        keypts_local << 0, 0,
                        0, 0,
                        1, 0;
        Mat3Xd keypts_world = mathutils::homo_to_real_coord<double>(GetSimilarityTransform() * mathutils::real_to_homo_coord<double>(keypts_local));
        return keypts_world;
    }

    // rect: [topleft, bottomright]
    Vec4_d ProjectToImageRect(const ::g2o::SE3Quat& transform_cw, const Mat33_d& kk) const {
        Mat3Xd corners_3d_world = compute_3D_cylinder_vertices();
        Mat2Xd corners_2d = mathutils::homo_to_real_coord<double>(kk * mathutils::homo_to_real_coord<double>(transform_cw.to_homogeneous_matrix() * mathutils::real_to_homo_coord<double>(corners_3d_world)));
        Vec2_d bottomright = corners_2d.rowwise().maxCoeff();
        Vec2_d topleft = corners_2d.rowwise().minCoeff();
        return Vec4_d(topleft(0), topleft(1), bottomright(0), bottomright(1));
    }

    // bbox: [X, Y, width, height]
    Vec4_d ProjectToImageBbox(const ::g2o::SE3Quat& transform_cw, const Mat33_d& kk) const
    {
        Vec4_d rect = ProjectToImageRect(transform_cw, kk);
        Vec2_d rect_center = (rect.head<2>() + rect.tail<2>()) / 2;
        Vec2_d wh = rect.tail<2>() - rect.head<2>();
        return Vec4_d(rect_center(0), rect_center(1), wh(0), wh(1));
    }

    // [[bbox], keypt1_x]
    Vec6_d ProjectToImageBboxAndKeypts(const ::g2o::SE3Quat& transform_cw, const Mat33_d& kk) const
    {
        Vec4_d bbox_proj = ProjectToImageBbox(transform_cw, kk);
        Mat3Xd keypts_3d_world = compute_keypt_3D();
        Mat2Xd keypys_2d = mathutils::homo_to_real_coord<double>(kk * mathutils::homo_to_real_coord<double>(transform_cw.to_homogeneous_matrix() * mathutils::real_to_homo_coord<double>(keypts_3d_world)));
        Vec6_d bboxAndKeypts;
        bboxAndKeypts << bbox_proj(0), bbox_proj(1), bbox_proj(2), bbox_proj(3), keypys_2d(0, 0), keypys_2d(1, 0);
        return bboxAndKeypts;
    }

    // [keypt1_x, keypt2_x]
    Vec4_d ProjectToImageKeypts(const ::g2o::SE3Quat& transform_cw, const Mat33_d& kk) const
    {
        Vec4_d bbox_proj = ProjectToImageBbox(transform_cw, kk);
        Mat3Xd keypts_3d_world = compute_keypts_3D();
        Mat2Xd keypys_2d = mathutils::homo_to_real_coord<double>(kk * mathutils::homo_to_real_coord<double>(transform_cw.to_homogeneous_matrix() * mathutils::real_to_homo_coord<double>(keypts_3d_world)));
        Vec4_d keypts;
        keypts << keypys_2d(0, 0), keypys_2d(1, 0), keypys_2d(0, 1), keypys_2d(1, 1);
        return keypts;
    }

};

class VertexLandmarkCylinder final : public ::g2o::BaseVertex<6, CylinderTarget>  // Note: 6 means the twist update is 6 digits (omega + upsilon). 
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexLandmarkCylinder()
    :BaseVertex<6, CylinderTarget>()
    {
        halfSizes.setZero();
        keypt1InLocal.setZero();
    }

    void initialize(const CylinderTarget& cylinder) {
        _estimate = cylinder;
        halfSizes = cylinder.halfSizes;
        keypt1InLocal = cylinder.keypt1InLocal;
    }

    void setToOriginImpl() override{
        std::cout <<"+++++++++++++++++++++++++++++++++" <<std::endl;
        std::cout <<"ShotVertex setToOriginImpl called" <<std::endl;
        std::cout <<"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" <<std::endl;
        _estimate = CylinderTarget();
        _estimate.initialize(halfSizes, keypt1InLocal);
    }

    void oplusImpl(const double *update) override;

    bool read(std::istream& is) override {return true;};
    bool write(std::ostream& os) const override {return os.good();};

    Vec3_d halfSizes;  // x, y size are same.
    Vec3_d keypt1InLocal;
};

class EdgeSE3CylinderProj : public ::g2o::BaseBinaryEdge<5, Vec5_d, VertexLandmarkCylinder, ShotVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef std::shared_ptr<EdgeSE3CylinderProj> Ptr;

    EdgeSE3CylinderProj(const unsigned int edgeId)
    :_edgeId(edgeId), BaseBinaryEdge<5, Vec5_d, VertexLandmarkCylinder, ShotVertex>()
    {}

    bool read(std::istream& is) override {return true;};
    bool write(std::ostream& os) const override {return os.good();};

    void computeError() override;
    double get_error_norm();

    bool depth_is_positive() const {
        const auto v1 = static_cast<const ShotVertex*>(_vertices.at(1));
        const auto v2 = static_cast<const VertexLandmarkCylinder*>(_vertices.at(0));
        return 0 < (v1->estimate().map(v2->estimate().poseInWorld.translation()))(2);
    }

    // void linearizeOplus() override;
    unsigned int _edgeId;
    Mat33_d kk;
    double focal_x_baseline_;
};


// !Note: don't change function interfaces. As it will be passed to g2o optimizer.
class StereoPerspectiveReprojEdge final : public ::g2o::BaseBinaryEdge<3, Vec3_d, VertexLandmarkCylinder, ShotVertex> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    StereoPerspectiveReprojEdge(const unsigned int edgeId)
        : _edgeId(edgeId), BaseBinaryEdge<3, Vec3_d, VertexLandmarkCylinder, ShotVertex>() {}

    ~StereoPerspectiveReprojEdge(){
        // Note: First all the member variables will be destructed in reverse order. Then this destructor will be called.
        // std::cout << "destructing edge (stereo):" << _edgeId << std::endl;
    }

    bool read(std::istream& is) override;

    bool write(std::ostream& os) const override;

    void computeError() override {
        const auto v1 = static_cast<const ShotVertex*>(_vertices.at(1));
        const auto v2 = static_cast<const VertexLandmarkCylinder*>(_vertices.at(0));
        const Vec3_d obs(_measurement);
        auto obsUpdate = cam_project(v1->estimate().map(v2->estimate().poseInWorld.translation()));
        _error = obs - obsUpdate;
    }

    void linearizeOplus() override;

    bool depth_is_positive() const {
        const auto v1 = static_cast<const ShotVertex*>(_vertices.at(1));
        const auto v2 = static_cast<const VertexLandmarkCylinder*>(_vertices.at(0));
        return 0 < (v1->estimate().map(v2->estimate().poseInWorld.translation()))(2);
    }

    inline Vec3_d cam_project(const Vec3_d& pos_c) const {
        const double reproj_x = fx_ * pos_c(0) / pos_c(2) + cx_;
        return {reproj_x, fy_ * pos_c(1) / pos_c(2) + cy_, reproj_x - focal_x_baseline_ / pos_c(2)};
    }

    double fx_, fy_, cx_, cy_, focal_x_baseline_;
    unsigned int _edgeId;
};

template<typename EdgeType>
class ReprojEdgeWrapper {
public:
    ReprojEdgeWrapper() = delete;  // Note: delete default constructor.

    ReprojEdgeWrapper(
        const unsigned int edgeId,
        std::shared_ptr<KeyFrame> pShot,
        ShotVertex* pShotVtx,
        std::shared_ptr<LandMark> pLm,
        VertexLandmarkCylinder* pLmVtx,
        const float refObjX,
        const float refObjY,
        const float refObjX_right,
        const float sqrt_chi_sq,
        const bool bUseHuberLoss
    );

    ReprojEdgeWrapper(
        const unsigned int edgeId,
        std::shared_ptr<KeyFrame> pShot,
        ShotVertex* pShotVtx,
        std::shared_ptr<LandMark> pLm,
        VertexLandmarkCylinder* pLmVtx,
        const Vec4_d& leftBbox,
        const Vec2_d& leftKeypts,
        const Vec4_d& rightBbox,
        const float sqrt_chi_sq,
        const bool bUseHuberLoss,
        const Mat55_d& infoMat
    );

    inline bool IsInlier() const {
        return _pEdge->level() == 0;
    }

    inline bool IsOutlier() const {
        return _pEdge->level() != 0;
    }

    inline void SetAsInlier() const {
        _pEdge->setLevel(0);
    }

    inline void SetAsOutlier() const {
        _pEdge->setLevel(1);
    }

    inline bool IsDepthPositive() const;

    std::shared_ptr<camera::CameraBase> _pCamera;
    std::shared_ptr<KeyFrame> _pShot;
    ::g2o::OptimizableGraph::Edge* _pEdge;
    std::shared_ptr<LandMark> _pLm;
    unsigned int _edgeId;
};

template<typename EdgeType>
ReprojEdgeWrapper<EdgeType>::ReprojEdgeWrapper(
    const unsigned int edgeId,
    std::shared_ptr<KeyFrame> pShot,
    ShotVertex* pShotVtx,
    std::shared_ptr<LandMark> pLm,
    VertexLandmarkCylinder* pLmVtx,
    const float refObjX,
    const float refObjY,
    const float refObjX_right,
    const float sqrt_chi_sq,
    const bool bUseHuberLoss
): _edgeId(edgeId), _pCamera(pShot->_pCamera), _pShot(pShot), _pLm(pLm) {
    EdgeType* edge = new EdgeType(edgeId);

    const Vec3_d obs{refObjX, refObjY, refObjX_right};
    edge->setMeasurement(obs);
    edge->setInformation(Mat33_d::Identity()); // * inv_sigma_sq);  // Note: no octave in object slam.

    edge->fx_ = _pCamera->_kk(0, 0);
    edge->fy_ = _pCamera->_kk(1, 1);
    edge->cx_ = _pCamera->_kk(0, 2);
    edge->cy_ = _pCamera->_kk(1, 2);
    edge->focal_x_baseline_ = _pCamera->_kk(0, 0) * _pCamera->_baseline;

    edge->setVertex(0, pLmVtx);
    edge->setVertex(1, pShotVtx);

    _pEdge = edge;

    // loss functionを設定
    if (bUseHuberLoss) {
        auto huber_kernel = new ::g2o::RobustKernelHuber();
        huber_kernel->setDelta(sqrt_chi_sq);
        _pEdge->setRobustKernel(huber_kernel);
    }
}

template<typename EdgeType>
ReprojEdgeWrapper<EdgeType>::ReprojEdgeWrapper(
    const unsigned int edgeId,
    std::shared_ptr<KeyFrame> pShot,
    ShotVertex* pShotVtx,
    std::shared_ptr<LandMark> pLm,
    VertexLandmarkCylinder* pLmVtx,
    const Vec4_d& leftBbox,
    const Vec2_d& leftKeypts,
    const Vec4_d& rightBbox,
    const float sqrt_chi_sq,
    const bool bUseHuberLoss,
    const Mat55_d& infoMat
): _edgeId(edgeId), _pCamera(pShot->_pCamera), _pShot(pShot), _pLm(pLm) {
    EdgeType* edge = new EdgeType(edgeId);
    Vec5_d obs;
    obs << leftKeypts(0), leftKeypts(1), leftBbox(0), leftBbox(1), rightBbox(0);
    edge->setMeasurement(obs);
    edge->setInformation(infoMat);

    edge->kk = _pCamera->_kk.cast<double>();
    edge->focal_x_baseline_ = _pCamera->_kk(0, 0) * _pCamera->_baseline;

    edge->setVertex(0, pLmVtx);
    edge->setVertex(1, pShotVtx);

    _pEdge = edge;

    // loss functionを設定
    if (bUseHuberLoss) {
        auto huber_kernel = new ::g2o::RobustKernelHuber();
        huber_kernel->setDelta(sqrt_chi_sq);
        _pEdge->setRobustKernel(huber_kernel);
    }
}

template<typename EdgeType>
bool ReprojEdgeWrapper<EdgeType>::IsDepthPositive() const {
    return static_cast<EdgeType*>(_pEdge)->EdgeType::depth_is_positive();
}

ShotVertex* CreateShotVertex(const unsigned int vtxId, const Mat44_d& worldToShotTransform, const bool isConstant);
VertexLandmarkCylinder* CreateLandmarkCylinderVertex(const unsigned int vtxId, const Mat44_d& poseInWorld, const Vec3_d& cylinderHalfSizes, const Vec3_d& keypt1InLocal, const bool isConstant);

}  // end of namespace g2outils
}  // end of namspace optimize


}  // end of namespace eventobjectslam

#endif