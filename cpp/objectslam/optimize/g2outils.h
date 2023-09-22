#ifndef EVENTOBJECTSLAM_OPTIMIZE_G2OUTILS_H
#define EVENTOBJECTSLAM_OPTIMIZE_G2OUTILS_H

#include "objectslam.h"
#include "landmark.h"

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
        std::cout << "destructing edge (stereo):" << _edgeId << std::endl;
    }

    bool read(std::istream& is) override;

    bool write(std::ostream& os) const override;

    void computeError() override {
        const auto v1 = static_cast<const ShotVertex*>(_vertices.at(1));
        const auto v2 = static_cast<const LandmarkPointVertex*>(_vertices.at(0));
        const Vec3_d obs(_measurement);
        _error = obs - cam_project(v1->estimate().map(v2->estimate()));
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

template<typename T>
class ReprojEdgeWrapper {
public:
    ReprojEdgeWrapper() = delete;  // Note: delete default constructor.

    ReprojEdgeWrapper(const unsigned int edgeId, std::shared_ptr<T> pShot, ShotVertex* pShotVtx,
                        std::shared_ptr<LandMark> pLm, LandmarkPointVertex* pLmVtx,
                        const float refObjX, const float refObjY, const float refObjX_right,
                        const float sqrt_chi_sq, const bool bUseHuberLoss = true);  // Note: sqrt_chi_sq is a strictiness threshold for huber kernel

    ~ReprojEdgeWrapper();

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
ReprojEdgeWrapper<T>::~ReprojEdgeWrapper(){
    std::cout << "destructing edge:" << _edgeId << std::endl;
}

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
    edge->focal_x_baseline_ = _pCamera->_baseline;

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
bool ReprojEdgeWrapper<T>::IsDepthPositive() const {
    return static_cast<StereoPerspectiveReprojEdge*>(_pEdge)->StereoPerspectiveReprojEdge::depth_is_positive();
}

ShotVertex* CreateShotVertex(const unsigned int vtxId, const Mat44_d& worldToShotTransform, const bool isConstant);
LandmarkPointVertex* CreateLandmarkPointVertex(const unsigned int vtxId, const Vec3_d& posInWorld, const bool isConstant);

}  // end of namespace g2outils
}  // end of namspace optimize


}  // end of namespace eventobjectslam

#endif