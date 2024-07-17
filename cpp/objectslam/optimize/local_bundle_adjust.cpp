#include "optimize/local_bundle_adjust.h"
#include "landmark.h"
#include "optimize/g2outils.h"
#include "mapdatabase.h"

#include <unordered_map>

#include <Eigen/StdVector>

#include <logging.h>
TDO_LOGGER("objectslam.optimize.localBA")

// #define USE_KEYFRAMEUPDATEMODE_CURRENTONLY
#define USE_KEYFRAMEUPDATEMODE_COVISIBILITY

namespace eventobjectslam {

void optimize::DoLocalBA(std::shared_ptr<KeyFrame> pCurrKeyframe, bool* const bForceStopFlag, const size_t numFirstIter, const size_t numSecondIter) {
    // -------- (1) --------
    // collect local/fixed keyframes, local landmark.

    // collect local keyframes of the current keyframe.
    std::unordered_map<unsigned int, std::shared_ptr<KeyFrame>> ids_localKeyframes;  // Note: keyframes that are within covisibilities of current keyframe.
    ids_localKeyframes[pCurrKeyframe->_keyFrameID] = pCurrKeyframe;
#ifdef USE_KEYFRAMEUPDATEMODE_COVISIBILITY
    const auto currCovisibilities = pCurrKeyframe->GetOrderedCovisibilities();
    TDO_LOG_DEBUG_FORMAT("num of covisibilities of keyframe(%d): %d", pCurrKeyframe->_keyFrameID % currCovisibilities.size());
    for (auto pLocalKeyFrame : currCovisibilities) {
        if (!pLocalKeyFrame) {
            TDO_LOG_ERROR("got empty pkeyframe, something is super wrong!");
            continue;
        }
        if (pLocalKeyFrame->IsToDelete()){
            continue;
        }
        if (pLocalKeyFrame->_keyFrameID == 0) {
            // do not optimize the first keyframe, which is the first frame as well.
            continue;
        }
        ids_localKeyframes[pLocalKeyFrame->_keyFrameID] = pLocalKeyFrame;
    }
#endif

    // collect local landmarks seen in local keyframes.
    std::unordered_map<unsigned int, std::shared_ptr<LandMark>> ids_localLandmarks;
    for (auto id_pLocalKeyframe : ids_localKeyframes) {
        const std::map<std::shared_ptr<LandMark>, unsigned int> landmarks_indices = id_pLocalKeyframe.second->GetObservedLandmarks();
        for (auto& localLandmark_index : landmarks_indices) {
            if (!localLandmark_index.first) {
                TDO_LOG_ERROR("got empty plandmark, something is super wrong!");
                continue;
            }

            if (localLandmark_index.first->IsToDelete()) {
                continue;
            }

            // avoid repeat
            if (ids_localLandmarks.count(localLandmark_index.first->_landmarkID)) {
                continue;
            }
            ids_localLandmarks[localLandmark_index.first->_landmarkID] = localLandmark_index.first;

        }
    }

    // collect fixed keyframes: the first keyframe, or keyframes that observe local landmarks but NOT in local keyframes.
    std::unordered_map<unsigned int, std::shared_ptr<KeyFrame>> ids_fixedKeyframes;
    for (auto id_localLandmark : ids_localLandmarks) {
        const std::map<std::shared_ptr<KeyFrame>, unsigned int> observations_indices = id_localLandmark.second->GetObservations();
        for (auto& pKeyframe_index : observations_indices) {
            auto pFixedKeyframe = pKeyframe_index.first;
            if (!pFixedKeyframe) {
                TDO_LOG_ERROR("got empty pkeyframe, something is super wrong!");
                continue;
            }
            if (pFixedKeyframe->IsToDelete()){
                continue;
            }
            // don't add keyframes that belong to local keyframes.
            if (ids_localKeyframes.count(pFixedKeyframe->_keyFrameID)) {
                continue;
            }
            // avoid repeat
            if (ids_fixedKeyframes.count(pFixedKeyframe->_keyFrameID)) {
                continue;
            }
            ids_fixedKeyframes[pFixedKeyframe->_keyFrameID] = pFixedKeyframe;
        }
    }

    // -------- (2) --------
    // build optimizer
    // // case: simple map point reprojection
    // auto linearSolver = std::make_unique<::g2o::LinearSolverCSparse<::g2o::BlockSolverPL<6, 3>::PoseMatrixType>>();  // Note: here ::g2o means a global namespace from outside of eventobjectslam.
    // auto blockSolver = std::make_unique<::g2o::BlockSolverPL<6, 3>>(std::move(linearSolver));
    // auto algorithm = new ::g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));

    // case: self-defined vertex
    std::unique_ptr<::g2o::BlockSolverX::LinearSolverType> linearSolver = std::make_unique<::g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>>();
    std::unique_ptr<::g2o::BlockSolverX> blockSolver = std::make_unique<::g2o::BlockSolverX>(std::move(linearSolver));
    ::g2o::OptimizationAlgorithmLevenberg* algorithm = new ::g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));

    ::g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(algorithm);

    if (bForceStopFlag) {
        optimizer.setForceStopFlag(bForceStopFlag);
    }

    // -------- (3) --------
    // create g2o vertex from keyframes and set to optimizer.
    std::unordered_map<unsigned int, g2outils::ShotVertex*> ids_keyfrmVtx;
    ids_keyfrmVtx.reserve(ids_localKeyframes.size() + ids_fixedKeyframes.size());
    // pre-save the keyframes that are used in g2o vertex
    std::unordered_map<unsigned int, std::shared_ptr<KeyFrame>> allKeyframes;
    unsigned int maxKeyframeID = 0;
    // set local keyframes to the optimizer
    for (auto& id_pLocalKeyframe : ids_localKeyframes) {
        auto pLocalKeyframe = id_pLocalKeyframe.second;
        // if (pLocalKeyframe->IsToDelete()) {
        //     continue;
        // }
        allKeyframes.emplace(id_pLocalKeyframe);
        TDO_LOG_CRITICAL("pLocalKeyframe->_keyFrameID:" << std::to_string(pLocalKeyframe->_keyFrameID) << ", isToDelte: " << (pLocalKeyframe->IsToDelete()? "true" : "false"));
        auto pKeyfrmVtx = g2outils::CreateShotVertex(pLocalKeyframe->_keyFrameID, pLocalKeyframe->GetKeyframePoseInWorld().inverse().cast<double>(), pLocalKeyframe->_keyFrameID==0?true:false);
        ids_keyfrmVtx[pLocalKeyframe->_keyFrameID] = pKeyfrmVtx;
        optimizer.addVertex(pKeyfrmVtx);
        maxKeyframeID = maxKeyframeID > pLocalKeyframe->_keyFrameID ? maxKeyframeID : pLocalKeyframe->_keyFrameID;
    }

    // set fixed keyframes to optimizer
    for (auto& id_pFixedKeyframe : ids_fixedKeyframes) {
        auto pFixedKeyframe = id_pFixedKeyframe.second;
        // if (pFixedKeyframe->IsToDelete()) {
        //     continue;
        // }
        allKeyframes.emplace(id_pFixedKeyframe);
        TDO_LOG_CRITICAL("pLocalKeyframe->_keyFrameID:" << std::to_string(pFixedKeyframe->_keyFrameID) << ", isToDelte: " << (pFixedKeyframe->IsToDelete()? "true" : "false"));
        auto pKeyfrmVtx = g2outils::CreateShotVertex(pFixedKeyframe->_keyFrameID, pFixedKeyframe->GetKeyframePoseInWorld().inverse().cast<double>(), true);
        ids_keyfrmVtx[pFixedKeyframe->_keyFrameID] = pKeyfrmVtx;
        optimizer.addVertex(pKeyfrmVtx);
        maxKeyframeID = maxKeyframeID > pFixedKeyframe->_keyFrameID ? maxKeyframeID : pFixedKeyframe->_keyFrameID;
    }

    // -------- (4) --------
    // Connect keyframe and landmark vertices by reprojection edge.
    std::unordered_map<unsigned int, g2outils::VertexLandmarkCylinder*> ids_landmarkPointVtx;
    ids_landmarkPointVtx.reserve(ids_localLandmarks.size());
    
    using ReprojEdgeWrapper = g2outils::ReprojEdgeWrapper<g2outils::StereoPerspectiveReprojEdge>;
    std::vector<ReprojEdgeWrapper> reprojEdgeWraps;
    reprojEdgeWraps.reserve(allKeyframes.size() * ids_localLandmarks.size());

    // 有意水準5%のカイ2乗値. for huber kernel. 自由度n=3
    constexpr float chi_sq_3D = 7.81473;
    const float sqrt_chi_sq_3D = std::sqrt(chi_sq_3D);
    unsigned int edgeId = 0;
    unsigned int ldmVertexId = 0;
    for (auto& id_localLm : ids_localLandmarks) {
        auto pLocalLm = id_localLm.second;
        // create g2o vertex from the landmark and set to optimizer.
        auto landmarkPoseInWorld = pLocalLm->GetLandmarkPoseInWorld().cast<double>();
        ldmVertexId = maxKeyframeID + 1 + pLocalLm->_landmarkID;
        Vec3_d cylinderHalfSizes = (pLocalLm->GetLandmarkSize() / 2).cast<double>();
        auto pLmVtx = g2outils::CreateLandmarkCylinderVertex(ldmVertexId, landmarkPoseInWorld, cylinderHalfSizes, pLocalLm->GetKeypt1InLandmark().cast<double>(), false);
        optimizer.addVertex(pLmVtx);
        ids_landmarkPointVtx[pLocalLm->_landmarkID] = pLmVtx;
        const std::map<std::shared_ptr<KeyFrame>, unsigned int> observations_indices = pLocalLm->GetObservations();
        for (const auto& pObs_idx : observations_indices) {
            auto pKeyframe = pObs_idx.first;
            auto idx = pObs_idx.second;
            if (!pKeyframe) {
                TDO_LOG_ERROR("got empty pkeyframe, the target keyframe might be deleted already!");
                continue;
            }
            if (!allKeyframes.count(pKeyframe->_keyFrameID)) {
                continue;
            }

            const auto pKeyfrmVtx = ids_keyfrmVtx[pKeyframe->_keyFrameID];
            const float centerX = pKeyframe->_refObjects[idx]->_detection._pLeftBbox->_centerX;
            const float centerY = pKeyframe->_refObjects[idx]->_detection._pLeftBbox->_centerY;
            const float centerXRight = pKeyframe->_refObjects[idx]->_detection._pRightBbox->_centerX;
            ReprojEdgeWrapper reprojEdgeWrap(edgeId, pKeyframe, pKeyfrmVtx, pLocalLm, pLmVtx, centerX, centerY, centerXRight, sqrt_chi_sq_3D, true);
            edgeId++;
            reprojEdgeWraps.push_back(reprojEdgeWrap);
            optimizer.addEdge(reprojEdgeWrap._pEdge);
            if (observations_indices.size() == 1){
                TDO_LOG_DEBUG_FORMAT("landmark(%d) has only one observation, disable the reprojEdgeWrap edge.", pLocalLm->_landmarkID);
                reprojEdgeWrap.SetAsOutlier();
            }
        }
    }

    using ReprojEdge2Wrapper = g2outils::ReprojEdgeWrapper<g2outils::EdgeSE3CylinderProj>;
    std::vector<ReprojEdge2Wrapper> reprojEdge2Wraps;
    reprojEdge2Wraps.reserve(allKeyframes.size() * ids_localLandmarks.size());

    // camera-object 2d measurement, including bbox and keypts
    double camera_object2d_BA_weight = 0.1;  // Note: weight in optimization.
    Vec5_d inv_sigma;
    inv_sigma.setOnes();
    inv_sigma *= camera_object2d_BA_weight;
    Mat55_d cam_object2d_sigma = inv_sigma.cwiseProduct(inv_sigma).asDiagonal();
    const float tHuberObject2d = std::sqrt(8.0);  // Note: 1000 object reprojection error is usually large.
    for (auto id_pLocalLm : ids_localLandmarks) {
        auto pLocalLm = id_pLocalLm.second;
        auto pLmVtx = ids_landmarkPointVtx[pLocalLm->_landmarkID];
        const std::map<std::shared_ptr<KeyFrame>, unsigned int> observations_indices = pLocalLm->GetObservations();
        for (const auto& pObs_idx : observations_indices) {
            auto pKeyframe = pObs_idx.first;
            auto idxRefObj = pObs_idx.second;
            if (!pKeyframe) {
                TDO_LOG_ERROR("got empty pkeyframe, the target keyframe might be deleted already!");
                continue;
            }
            if (!allKeyframes.count(pKeyframe->_keyFrameID)) {
                continue;
            }
            const auto pKeyfrmVtx = ids_keyfrmVtx[pKeyframe->_keyFrameID];
            Vec4_d leftBbox;
            auto& threeDetection = pKeyframe->_refObjects[idxRefObj]->_detection;
            leftBbox << threeDetection._pLeftBbox->_centerX, threeDetection._pLeftBbox->_centerY, threeDetection._pLeftBbox->_bWidth, threeDetection._pLeftBbox->_bHeight;
            Vec2_d leftKeypts;
            leftKeypts << threeDetection._pLeftBbox->_keypts[0](0), threeDetection._pLeftBbox->_keypts[0](1);
            Vec4_d rightBbox;
            rightBbox << threeDetection._pRightBbox->_centerX, threeDetection._pRightBbox->_centerY, threeDetection._pRightBbox->_bWidth, threeDetection._pRightBbox->_bHeight;
            Mat55_d infoMat = cam_object2d_sigma * threeDetection._detectionScore * threeDetection._detectionScore;
            ReprojEdge2Wrapper reprojEdge2Wrap(edgeId, pKeyframe, pKeyfrmVtx, pLocalLm, pLmVtx, leftBbox, leftKeypts, rightBbox, tHuberObject2d, true, infoMat);
            edgeId++;
            auto key_keyfrm_lm = std::make_pair(pKeyframe, pLocalLm);
            reprojEdge2Wraps.push_back(reprojEdge2Wrap);
            optimizer.addEdge(reprojEdge2Wrap._pEdge);
            if (observations_indices.size() == 1){
                TDO_LOG_DEBUG_FORMAT("landmark(%d) has only one observation, disable the reprojEdge2Wrap edge.", pLocalLm->_landmarkID);
                reprojEdge2Wrap.SetAsOutlier();
            }
        }
    }

    // -------- (5) --------
    // 1st round of optimization
    if (bForceStopFlag) {
        if (*bForceStopFlag) {
            return;
        }
    }

    optimizer.initializeOptimization();
    TDO_LOG_CRITICAL_FORMAT("start optimize with %d allKeyframes, %d ids_localLandmarks, %d ReprojEdges.", allKeyframes.size() % ids_localLandmarks.size() % reprojEdgeWraps.size());
    try{
        optimizer.optimize(numFirstIter);
    }
    catch (const std::exception& e){
        TDO_LOG_CRITICAL("caught exception during optimization. skip localBA. Error:\n" << e.what());
        return;
    }
    catch (...) {
        TDO_LOG_CRITICAL("caught unknown exception during optimization. skip localBA.");
        return;
    }

    // -------- (6) --------
    // remove outliers and 2nd round.
    if (*bForceStopFlag) {

        for (size_t ii = 0; ii < reprojEdgeWraps.size(); ii++) {
            auto& reprojEdgeWrap = reprojEdgeWraps[ii];
            auto pEdge = reprojEdgeWrap._pEdge;
            auto localLm = reprojEdgeWrap._pLm;
            if (localLm->IsToDelete()) {
                continue;
            }

            if (chi_sq_3D < pEdge->chi2() || !reprojEdgeWrap.IsDepthPositive()) {
                reprojEdgeWrap.SetAsOutlier();
            }

            pEdge->setRobustKernel(nullptr);
        }

        for (size_t ii = 0; ii < reprojEdge2Wraps.size(); ii++) {
            auto& reprojEdge2Wrap = reprojEdge2Wraps[ii];
            auto pEdge = reprojEdge2Wrap._pEdge;
            auto localLm = reprojEdge2Wrap._pLm;
            if (localLm->IsToDelete()) {
                continue;
            }

            if ((tHuberObject2d * tHuberObject2d) < pEdge->chi2() || !reprojEdge2Wrap.IsDepthPositive()) {
                reprojEdge2Wrap.SetAsOutlier();
            }

            pEdge->setRobustKernel(nullptr);
        }

        optimizer.initializeOptimization();
        optimizer.optimize(numSecondIter);
    }

    // -------- (7) --------
    // collect outliers
    std::vector<std::pair<std::shared_ptr<KeyFrame>, std::shared_ptr<LandMark>>> outlier_observations_lms;
    outlier_observations_lms.reserve(reprojEdgeWraps.size());
    for (auto& reprojEdgeWrap : reprojEdgeWraps) {
        auto pEdge = reprojEdgeWrap._pEdge;
        auto localLm = reprojEdgeWrap._pLm;
        if (localLm->IsToDelete()) {
            continue;
        }

        TDO_LOG_VERBOSE_FORMAT("Edge between keyframe(%d), landmark(%d), chi2 %f, depthPosi %s",
                                    reprojEdgeWrap._pShot->_keyFrameID
                                    % reprojEdgeWrap._pLm->_landmarkID
                                    % pEdge->chi2()
                                    % std::to_string(reprojEdgeWrap.IsDepthPositive()));
        if (chi_sq_3D < pEdge->chi2() || !reprojEdgeWrap.IsDepthPositive()) {
            outlier_observations_lms.push_back(std::make_pair(reprojEdgeWrap._pShot, reprojEdgeWrap._pLm));
        }
    }
    for (auto& reprojEdge2Wrap : reprojEdge2Wraps) {
        auto pEdge = reprojEdge2Wrap._pEdge;
        auto localLm = reprojEdge2Wrap._pLm;
        if (localLm->IsToDelete()) {
            continue;
        }

        TDO_LOG_VERBOSE_FORMAT("Edge2 between keyframe(%d), landmark(%d), chi2 %f, depthPosi %s",
                                    reprojEdge2Wrap._pShot->_keyFrameID
                                    % reprojEdge2Wrap._pLm->_landmarkID
                                    % pEdge->chi2()
                                    % std::to_string(reprojEdge2Wrap.IsDepthPositive()));
        if ((tHuberObject2d * tHuberObject2d) < pEdge->chi2() || !reprojEdge2Wrap.IsDepthPositive()) {
            TDO_LOG_CRITICAL_FORMAT("outlier by cylinder proj: chi2 %f, isDepth %d", pEdge->chi2() % reprojEdge2Wrap.IsDepthPositive());
            outlier_observations_lms.push_back(std::make_pair(reprojEdge2Wrap._pShot, reprojEdge2Wrap._pLm));
        }
    }

    // -------- (8) --------
    // update the map
    {
        std::lock_guard<std::mutex> lock(MapDataBase::_mtxDatabase);
        if (!outlier_observations_lms.empty()){
            for (auto& outlier_obs_lm : outlier_observations_lms) {
                auto pKeyfrm = outlier_obs_lm.first;
                auto pLm = outlier_obs_lm.second;
                TDO_LOG_CRITICAL_FORMAT("outlier pair keyframe(%d), landmark(%d)", pKeyfrm->_keyFrameID % pLm->_landmarkID);
                pKeyfrm->DeleteOneObservedLandmark(pLm);
                pLm->DeleteObservation(pKeyfrm);
            }
        }

        for (auto id_localKeyfrm : ids_localKeyframes) {
            auto pLocalKeyfrm = id_localKeyfrm.second;
            auto pKeyfrmVtx = ids_keyfrmVtx[pLocalKeyfrm->_keyFrameID];
            TDO_LOG_CRITICAL_FORMAT("updating pose of keyframe(%d)", pLocalKeyfrm->_keyFrameID);
            pLocalKeyfrm->SetKeyframePoseInWorld(pKeyfrmVtx->estimate().to_homogeneous_matrix().inverse().cast<float>());
        }

        for (auto id_localLm : ids_localLandmarks) {
            auto pLocalLm = id_localLm.second;
            auto pLmVtx = ids_landmarkPointVtx[pLocalLm->_landmarkID];
            Mat44_t landmarkPoseInWorld = pLocalLm->GetLandmarkPoseInWorld();
            landmarkPoseInWorld.block(0, 3, 3, 1) = pLmVtx->estimate().poseInWorld.translation().cast<float>();
            TDO_LOG_CRITICAL_FORMAT("updating pose of landmark(%d)", pLocalLm->_landmarkID);
            pLocalLm->SetLandmarkPoseInWorld(landmarkPoseInWorld);
        }

    }

    TDO_LOG_CRITICAL_FORMAT("localBA finished with %d keyfrm, %d landmarks updated. %d outlierPairs!", 
                                    ids_localKeyframes.size() %
                                    ids_localLandmarks.size() %
                                    outlier_observations_lms.size());

}

}  // end of namespace eventobjectslam