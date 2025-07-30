#ifndef EVENTOBJECTSLAM_OPTIMIZE_G2OUTILS_H
#define EVENTOBJECTSLAM_OPTIMIZE_G2OUTILS_H

#include "objectslam.h"
#include "landmark.h"
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
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/types/slam2d/se2.h>

#define USE_KEYFRAMEUPDATEMODE_CURRENTONLY
// #define USE_KEYFRAMEUPDATEMODE_COVISIBILITY

// #define DO_SE3

namespace eventobjectslam {

namespace optimize {

namespace g2outils {

class ShotVertex final : public ::g2o::BaseVertex<6, ::g2o::SE3Quat> {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ShotVertex()
    : BaseVertex<6, ::g2o::SE3Quat>() {}

    ~ShotVertex() = default;

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

class LandmarkPointVertex final : public ::g2o::BaseVertex<3, Vec3_d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    LandmarkPointVertex()
        : BaseVertex<3, Vec3_d>() {}

    ~LandmarkPointVertex() = default;

    bool read(std::istream& is) override;

    bool write(std::ostream& os) const override;

    void setToOriginImpl() override {
        _estimate.fill(0);
    }

    void oplusImpl(const double* update) override {
        Eigen::Map<const Vec3_d> v(update);
        _estimate += v;
    }

};

// !Note: don't change function interfaces. As it will be passed to g2o optimizer.
class StereoPerspectiveReprojEdge final : public ::g2o::BaseBinaryEdge<3, Vec3_d, LandmarkPointVertex, ShotVertex> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    StereoPerspectiveReprojEdge(const unsigned int edgeId)
        : _edgeId(edgeId), BaseBinaryEdge<3, Vec3_d, LandmarkPointVertex, ShotVertex>() {}

    ~StereoPerspectiveReprojEdge(){
        // Note: First all the member variables will be destructed in reverse order. Then this destructor will be called.
        // std::cout << "destructing edge (stereo):" << _edgeId << std::endl;
    }

    bool read(std::istream& is) override;

    bool write(std::ostream& os) const override;

    void computeError() override {
        const auto v1 = static_cast<const ShotVertex*>(_vertices.at(1));
        const auto v2 = static_cast<const LandmarkPointVertex*>(_vertices.at(0));
        const Vec3_d obs(_measurement);
        auto obsUpdate = cam_project(v1->estimate().map(v2->estimate()));
        _error = obs - obsUpdate;
        // std::cout << "edgeId: " << _edgeId << ", obs: {" << obs[0] << ", " << obs[1] << ", " << obs[2] << "}, update: {" << obsUpdate[0] << ", " << obsUpdate[1] << ", " << obsUpdate[2] << "}." << std::endl;
    }

    void linearizeOplus() override;

    bool depth_is_positive() const {
        const auto v1 = static_cast<const ShotVertex*>(_vertices.at(1));
        const auto v2 = static_cast<const LandmarkPointVertex*>(_vertices.at(0));
        return 0 < (v1->estimate().map(v2->estimate()))(2);
    }

    inline Vec3_d cam_project(const Vec3_d& pos_c) const {
        const double reproj_x = fx_ * pos_c(0) / pos_c(2) + cx_;
        return {reproj_x, fy_ * pos_c(1) / pos_c(2) + cy_, reproj_x - focal_x_baseline_ / pos_c(2)};
    }

    double fx_, fy_, cx_, cy_, focal_x_baseline_;
    unsigned int _edgeId;
};

// classes for SE2 optimization
class ShotVertexSE2 final : public ::g2o::BaseVertex<3, ::g2o::SE2> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ShotVertexSE2()
    : BaseVertex<3, ::g2o::SE2>() {}

    ~ShotVertexSE2(){}

    bool read(std::istream& is) override{
        double x, y, th;
        is >> x >> y >> th;
        setEstimate(g2o::SE2(x, y, th));
        return true;
    }

    bool write(std::ostream& os) const override {
        const Vec3_d v = _estimate.toVector();
        os << v.x() << " " << v.y() << " " << v.z();
        return os.good();
    }

    void oplusImpl(const ::g2o::number_t* update_) override {
        Vec3_d update(update_[0], update_[1], update_[2]);
        g2o::SE2 delta;
        delta.fromVector(update);
        _estimate = delta * _estimate;
    }

    void setToOriginImpl() override {
        _estimate = ::g2o::SE2();
    }
};

class LandmarkPointVertex2D final : public ::g2o::BaseVertex<2, Vec2_d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LandmarkPointVertex2D()
    : BaseVertex<2, Vec2_d>() {}

    ~LandmarkPointVertex2D() = default;

    bool read(std::istream& is) override {
        is >> _estimate[0] >> _estimate[1];
        return true;
    }

    bool write(std::ostream& os) const override {
        os << _estimate.x() << " " << _estimate.y();
        return os.good();
    }

    void setToOriginImpl() override {
        _estimate.setZero();
    }

    void oplusImpl(const double* update) override {
        _estimate[0] += update[0];
        _estimate[1] += update[1];
    }
};

class StereoPerspectiveReprojEdgeSE2 final : public ::g2o::BaseBinaryEdge<3, Vec3_d, LandmarkPointVertex2D, ShotVertexSE2> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    StereoPerspectiveReprojEdgeSE2(const unsigned int edgeId, const double Y_ws, const double Y_wp)  // Y_ws, is Y of camera in world; Y_wp is Y of point in world. These will not be optimized by BA.
    : _edgeId(edgeId), BaseBinaryEdge<3, Vec3_d, LandmarkPointVertex2D, ShotVertexSE2>()
    {
        _Y_wp = Y_wp;
        _Y_ws = Y_ws;
    }

    ~StereoPerspectiveReprojEdgeSE2() = default;

    bool read(std::istream& is) override {
        is >> _measurement[0] >> _measurement[1];
        for (int i = 0; i < information().rows() && is.good(); ++i) {
            for (int j = i; j < information().cols() && is.good(); ++j) {
            double v;
            is >> v;
            information()(i, j) = v;
            if (i != j) information()(j, i) = v;
            }
        }
        return true;
    }

    bool write(std::ostream& os) const override {
        os << _measurement.x() << " " << _measurement.y() << " ";
        for (int i = 0; i < information().rows(); ++i) {
            for (int j = i; j < information().cols(); ++j) {
            os << information()(i, j) << " ";
            }
        }
        return os.good();
    }

    void computeError() override {
        const ShotVertexSE2* vPose = static_cast<const ShotVertexSE2*>(_vertices.at(1));
        const LandmarkPointVertex2D* vPoint = static_cast<const LandmarkPointVertex2D*>(_vertices.at(0));
        const ::g2o::SE2& X = vPose->estimate(); // (X_sw, Z_sw, Ry_sw)
        const Eigen::Vector2d& p = vPoint->estimate();
        const double ct = std::cos(X.rotation().angle());
        const double st = std::sin(X.rotation().angle());
        Eigen::Matrix2d R;
        R << ct, -st,
             st,  ct;
        Vec2_d t(X.translation().x(), X.translation().y());
        // predicted measurement: z_hat = R^T (p - t) => R*p_cam+t=p_world
        Vec2_d p_cam = R.transpose() * (p - t);  // p_cam is (X, Z) in the camera frame.
        const Vec3_d obs(_measurement);  // measurement is (x_l, y_l, x_r) in pixel coords.
        Vec3_d obsUpdate = cam_project(p_cam);
        _error = obs - obsUpdate;
    }

    inline Vec3_d cam_project(const Vec2_d& pos_xz) const {
        // pos_xz is (X, Z) in the camera frame.
        const Vec3_d pos_xyz(pos_xz(0), _Y_wp - _Y_ws, pos_xz(1));
        const double reproj_x = fx_ * pos_xyz(0) / pos_xyz(2) + cx_;
        const double reproj_y = fy_ * pos_xyz(1) / pos_xyz(2) + cy_;
        const double reproj_x_r = reproj_x - focal_x_baseline_ / pos_xyz(2);
        return {reproj_x, reproj_y, reproj_x_r};
    }

    bool depth_is_positive() const {
        const ShotVertexSE2* vPose = static_cast<const ShotVertexSE2*>(_vertices.at(1));
        const LandmarkPointVertex2D* vPoint = static_cast<const LandmarkPointVertex2D*>(_vertices.at(0));
        const ::g2o::SE2& X = vPose->estimate(); // (X_sw, Z_sw, Ry_sw)
        const Eigen::Vector2d& p = vPoint->estimate();
        const double ct = std::cos(X.rotation().angle());
        const double st = std::sin(X.rotation().angle());
        Eigen::Matrix2d R;
        R << ct, -st,
             st,  ct;
        Vec2_d t(X.translation().x(), X.translation().y());
        Vec2_d p_cam = R.transpose() * (p - t);
        return p_cam[1] > 0;
    }

    void linearizeOplus() override;

    double _Y_ws, _Y_wp;  // Y_ws, is Y of T_ws; Y_wp is Y of T_wp. These will not be optimized by BA.
    double fx_, fy_, cx_, cy_, focal_x_baseline_;  // camera params will not be optimized by BA.
    unsigned int _edgeId;

};


/* Edge wrapper and utility functions */
template<typename T>
class ReprojEdgeWrapper {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ReprojEdgeWrapper() = delete;  // Note: delete default constructor.

    ReprojEdgeWrapper(const unsigned int edgeId, std::shared_ptr<T> pShot, ShotVertex* pShotVtx,
                        std::shared_ptr<LandMark> pLm, LandmarkPointVertex* pLmVtx,
                        const float refObjX, const float refObjY, const float refObjX_right,
                        const float sqrt_chi_sq, const bool bUseHuberLoss = true);  // Note: sqrt_chi_sq is a strictiness threshold for huber kernel

    ReprojEdgeWrapper(const unsigned int edgeId, std::shared_ptr<T> pShot, ShotVertexSE2* pShotVtx,
                        std::shared_ptr<LandMark> pLm, LandmarkPointVertex2D* pLmVtx,
                        const float refObjX, const float refObjY, const float refObjX_right,
                        const float Y_ws, const float Y_wp,
                        const float sqrt_chi_sq, const bool bUseHuberLoss = true);    

    ~ReprojEdgeWrapper() = default;

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
    std::shared_ptr<T> _pShot;
    ::g2o::OptimizableGraph::Edge* _pEdge;
    std::shared_ptr<LandMark> _pLm;
    unsigned int _edgeId;
};

template<typename T>
ReprojEdgeWrapper<T>::ReprojEdgeWrapper(const unsigned int edgeId, std::shared_ptr<T> pShot, ShotVertex* pShotVtx,
                                        std::shared_ptr<LandMark> pLm, LandmarkPointVertex* pLmVtx,
                                        const float refObjX, const float refObjY, const float refObjX_right,
                                        const float sqrt_chi_sq, const bool bUseHuberLoss)
    : _edgeId(edgeId), _pCamera(pShot->_pCamera), _pShot(pShot), _pLm(pLm) {
    StereoPerspectiveReprojEdge* edge = new StereoPerspectiveReprojEdge(edgeId);

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


template<typename T>
ReprojEdgeWrapper<T>::ReprojEdgeWrapper(const unsigned int edgeId, std::shared_ptr<T> pShot, ShotVertexSE2* pShotVtx,
                                        std::shared_ptr<LandMark> pLm, LandmarkPointVertex2D* pLmVtx,
                                        const float refObjX, const float refObjY, const float refObjX_right,
                                        const float Y_ws, const float Y_wp,
                                        const float sqrt_chi_sq, const bool bUseHuberLoss)
    : _edgeId(edgeId), _pCamera(pShot->_pCamera), _pShot(pShot), _pLm(pLm) {
    StereoPerspectiveReprojEdgeSE2* edge = new StereoPerspectiveReprojEdgeSE2(edgeId, Y_ws, Y_wp);

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


#ifdef DO_SE3
    template<typename T>
    bool ReprojEdgeWrapper<T>::IsDepthPositive() const {
        return static_cast<StereoPerspectiveReprojEdge*>(_pEdge)->StereoPerspectiveReprojEdge::depth_is_positive();
    }
#else
    template<typename T>
    bool ReprojEdgeWrapper<T>::IsDepthPositive() const {
        return static_cast<StereoPerspectiveReprojEdgeSE2*>(_pEdge)->StereoPerspectiveReprojEdgeSE2::depth_is_positive();
    }
#endif

ShotVertex* CreateShotVertex(const unsigned int vtxId, const Mat44_d& worldToShotTransform, const bool isConstant);
LandmarkPointVertex* CreateLandmarkPointVertex(const unsigned int vtxId, const Vec3_d& posInWorld, const bool isConstant);

ShotVertexSE2* CreateShotVertexSE2(const unsigned int vtxId, const Mat44_d& worldToShotTransform, const bool isConstant);
LandmarkPointVertex2D* CreateLandmarkPointVertex2D(const unsigned int vtxId, const Vec3_d& posInWorld, const bool isConstant);

const Mat44_t ReconstructNewCameraPoseInWorld(ShotVertexSE2* pShotVtx, const Mat44_t& oldCameraPoseInWorld);
const Vec3_t ReconstructNewPointInWorld(LandmarkPointVertex2D* pLmVtx, const Vec3_t& oldLmPointInWorld);
}  // end of namespace g2outils

}  // end of namspace optimize


}  // end of namespace eventobjectslam

#endif