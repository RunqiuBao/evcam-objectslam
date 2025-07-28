#include "optimize/local_bundle_adjust.h"
#include "landmark.h"
#include "optimize/g2outils.h"
#include "mapdatabase.h"
#include "mathutils.h"

#include <unordered_map>

#include <Eigen/StdVector>

#include <logging.h>
TDO_LOGGER("objectslam.optimize.localBA")

#define USE_KEYFRAMEUPDATEMODE_CURRENTONLY
// #define USE_KEYFRAMEUPDATEMODE_COVISIBILITY

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
            continue;  // Note: skip the first keyframe.
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
    if (ids_localLandmarks.size() == 0) {
        TDO_LOG_ERROR_FORMAT("No local landmarks found. Current keyframe is the first keyframe: %d", pCurrKeyframe->_keyFrameID);
        const std::map<std::shared_ptr<LandMark>, unsigned int> landmarks_indices = pCurrKeyframe->GetObservedLandmarks();
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
    auto linearSolver = std::make_unique<::g2o::LinearSolverCSparse<::g2o::BlockSolver_6_3::PoseMatrixType>>();  // Note: here ::g2o means a global namespace from outside of eventobjectslam.
    auto blockSolver = std::make_unique<::g2o::BlockSolver_6_3>(std::move(linearSolver));
    auto algorithm = new ::g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));

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
        allKeyframes.emplace(id_pLocalKeyframe);
        auto pKeyfrmVtx = g2outils::CreateShotVertex(pLocalKeyframe->_keyFrameID, pLocalKeyframe->GetKeyframePoseInWorld().inverse().cast<double>(), false);
        ids_keyfrmVtx[pLocalKeyframe->_keyFrameID] = pKeyfrmVtx;
        optimizer.addVertex(pKeyfrmVtx);
        maxKeyframeID = maxKeyframeID > pLocalKeyframe->_keyFrameID ? maxKeyframeID : pLocalKeyframe->_keyFrameID;
    }

    // set fixed keyframes to optimizer
    for (auto& id_pFixedKeyframe : ids_fixedKeyframes) {
        auto pFixedKeyframe = id_pFixedKeyframe.second;
        allKeyframes.emplace(id_pFixedKeyframe);
        auto pKeyfrmVtx = g2outils::CreateShotVertex(pFixedKeyframe->_keyFrameID, pFixedKeyframe->GetKeyframePoseInWorld().inverse().cast<double>(), true);
        ids_keyfrmVtx[pFixedKeyframe->_keyFrameID] = pKeyfrmVtx;
        optimizer.addVertex(pKeyfrmVtx);
        maxKeyframeID = maxKeyframeID > pFixedKeyframe->_keyFrameID ? maxKeyframeID : pFixedKeyframe->_keyFrameID;
    }

    // -------- (4) --------
    // Connect keyframe and landmark vertices by reprojection edge.
    std::unordered_map<unsigned int, g2outils::LandmarkPointVertex*> ids_landmarkPointVtx;
    ids_landmarkPointVtx.reserve(ids_localLandmarks.size());
    std::unordered_map<unsigned int, std::vector<g2outils::LandmarkPointVertex*>> ids_vec_landmarkPointKeyptVtx;
    
    using ReprojEdgeWrapper = g2outils::ReprojEdgeWrapper<KeyFrame>;
    std::vector<ReprojEdgeWrapper> reprojEdgeWraps;
    std::unordered_map<unsigned int, std::vector<ReprojEdgeWrapper>> ids_vec_reprojEdgeKptWraps;
    reprojEdgeWraps.reserve(allKeyframes.size() * ids_localLandmarks.size());

    // 有意水準5%のカイ2乗値. for huber kernel
    // 自由度n=3
    constexpr float chi_sq_3D = 7.81473;
    const float sqrt_chi_sq_3D = std::sqrt(chi_sq_3D);
    unsigned int edgeId = 0;
    unsigned int pointVertexId = 0;
    for (auto& id_localLm : ids_localLandmarks) {
        auto pLocalLm = id_localLm.second;
        // create g2o vertex from the landmark and set to optimizer.
        Mat44_t landmarkPoseInWorld = pLocalLm->GetLandmarkPoseInWorld(); 
        Vec3_t landmarkCenterInWorld = landmarkPoseInWorld.block(0, 3, 3, 1);
        auto pLmVtx = g2outils::CreateLandmarkPointVertex(pointVertexId, landmarkCenterInWorld.cast<double>(), false);
        pointVertexId++;
        optimizer.addVertex(pLmVtx);
        ids_landmarkPointVtx[pLocalLm->_landmarkID] = pLmVtx;
        int numKeypts = pLocalLm->_hasFacet?pLocalLm->GetVertices3DInLandmark().rows():1;
        std::vector<g2outils::LandmarkPointVertex*> vec_landmarkPointKeyptVtx;
        TDO_LOG_DEBUG_FORMAT("landmark(%d) center in world: %f, %f, %f",
                                pLocalLm->_landmarkID
                                % landmarkCenterInWorld(0)
                                % landmarkCenterInWorld(1)
                                % landmarkCenterInWorld(2));
        for (size_t indexKeypt=0; indexKeypt < numKeypts; indexKeypt++){
            Vec3_t keyptInWorld = pLocalLm->GetOneVertex3DInWorld(indexKeypt);
            auto pLmKeyptVtx = g2outils::CreateLandmarkPointVertex(pointVertexId, keyptInWorld.cast<double>(), false);
            pointVertexId++;
            optimizer.addVertex(pLmKeyptVtx);
            vec_landmarkPointKeyptVtx.push_back(pLmKeyptVtx);
            TDO_LOG_DEBUG_FORMAT("landmark(%d) keypt in world: %f, %f, %f",
                                    pLocalLm->_landmarkID
                                    % keyptInWorld(0)
                                    % keyptInWorld(1)
                                    % keyptInWorld(2));
        }
        ids_vec_landmarkPointKeyptVtx[pLocalLm->_landmarkID] = std::move(vec_landmarkPointKeyptVtx);

        const std::map<std::shared_ptr<KeyFrame>, unsigned int> observations_indices = pLocalLm->GetObservations();
        for (const auto& pObs_idx : observations_indices) {
            auto pKeyframe = pObs_idx.first;
            auto idx = pObs_idx.second;
            if (!pKeyframe) {
                TDO_LOG_ERROR("got empty pkeyframe, something is super wrong!");
                continue;
            }
            if (!allKeyframes.count(pKeyframe->_keyFrameID)) {
                continue;
            }

            const auto pKeyfrmVtx = ids_keyfrmVtx[pKeyframe->_keyFrameID];
            const float centerX = pKeyframe->_refObjects[idx]->_detection._pLeftBbox->_centerX;
            const float centerY = pKeyframe->_refObjects[idx]->_detection._pLeftBbox->_centerY;
            const float centerXRight = pKeyframe->_refObjects[idx]->_detection._pRightBbox->_centerX;
            ReprojEdgeWrapper reprojEdgeWrap(edgeId, pKeyframe, pKeyfrmVtx, pLocalLm, pLmVtx, centerX, centerY, centerXRight, sqrt_chi_sq_3D);
            reprojEdgeWraps.push_back(reprojEdgeWrap);
            optimizer.addEdge(reprojEdgeWrap._pEdge);
            TDO_LOG_DEBUG_FORMAT("center point reproj edge: keyframe(%d), landmark(%d), centerX %f, centerY %f, centerXRight %f",
                                    pKeyframe->_keyFrameID
                                    % pLocalLm->_landmarkID
                                    % centerX
                                    % centerY
                                    % centerXRight);

            for (size_t indexKeypt=0; indexKeypt < numKeypts; indexKeypt++){
                const float keyptX = pKeyframe->_refObjects[idx]->_detection._pLeftBbox->_keypts[indexKeypt](0);
                const float keyptY = pKeyframe->_refObjects[idx]->_detection._pLeftBbox->_keypts[indexKeypt](1);
                const float keyptXRight = pKeyframe->_refObjects[idx]->_detection._pRightBbox->_keypts[indexKeypt](0);
                auto& pLmKeyptVtx = ids_vec_landmarkPointKeyptVtx[pLocalLm->_landmarkID][indexKeypt];
                ReprojEdgeWrapper keyptReprojEdgeWrap(edgeId + reprojEdgeWraps.capacity(), pKeyframe, pKeyfrmVtx, pLocalLm, pLmKeyptVtx, keyptX, keyptY, keyptXRight, sqrt_chi_sq_3D);
                ids_vec_reprojEdgeKptWraps[pLocalLm->_landmarkID].push_back(keyptReprojEdgeWrap);
                optimizer.addEdge(keyptReprojEdgeWrap._pEdge);
                TDO_LOG_DEBUG_FORMAT("keypoint reproj edge: keyframe(%d), landmark(%d), keyptX %f, keyptY %f, keyptXRight %f",
                                        pKeyframe->_keyFrameID
                                        % pLocalLm->_landmarkID
                                        % keyptX
                                        % keyptY
                                        % keyptXRight);
            }

            edgeId++;
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
    int numReprojEdges = reprojEdgeWraps.size();
    std::unordered_map<unsigned int, std::vector<ReprojEdgeWrapper>>::const_iterator itREKW;
    for (itREKW = ids_vec_reprojEdgeKptWraps.begin(); itREKW != ids_vec_reprojEdgeKptWraps.end(); itREKW++) {
        numReprojEdges += itREKW->second.size();
    }
    TDO_LOG_CRITICAL_FORMAT("start optimize with %d allKeyframes, %d ids_localLandmarks, %d ReprojEdges.", allKeyframes.size() % ids_localLandmarks.size() % numReprojEdges);

    // if the keyframes are very close to each other, no need to do BA
    Vec3_t posCurrKeyframe = pCurrKeyframe->GetKeyframePoseInWorld().col(3).head<3>();
    float maxDistance = 0.0;
    std::unordered_map<unsigned int, std::shared_ptr<KeyFrame>>::const_iterator itKeyfrms = allKeyframes.begin();
    for (itKeyfrms++; itKeyfrms != allKeyframes.end(); itKeyfrms++) {
        Vec3_t posNextKeyframe = itKeyfrms->second->GetKeyframePoseInWorld().col(3).head<3>();
        float distance = (posNextKeyframe - posCurrKeyframe).norm();
        if (distance > maxDistance) {
            maxDistance = distance;
        }
    }
    if (maxDistance < 0.5) {
        TDO_LOG_WARN("keyframes are too close, skip localBA.");
        return;
    }

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
        for (auto& reprojEdgeWrap : reprojEdgeWraps) {
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

        for (itREKW = ids_vec_reprojEdgeKptWraps.begin(); itREKW != ids_vec_reprojEdgeKptWraps.end(); itREKW++){
            for (auto& reprojEdgeKptWrap : itREKW->second) {
                auto pEdge = reprojEdgeKptWrap._pEdge;
                auto localLm = reprojEdgeKptWrap._pLm;
                if (localLm->IsToDelete()) {
                    continue;
                }
    
                if (chi_sq_3D < pEdge->chi2() || !reprojEdgeKptWrap.IsDepthPositive()) {
                    reprojEdgeKptWrap.SetAsOutlier();
                }
    
                pEdge->setRobustKernel(nullptr);
            }
        }

        optimizer.initializeOptimization();
        optimizer.optimize(numSecondIter);
    }

    // -------- (7) --------
    // collect outliers
    std::vector<std::pair<std::shared_ptr<KeyFrame>, std::shared_ptr<LandMark>>> outlier_observations_lms;
    outlier_observations_lms.reserve(numReprojEdges);
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
    for (itREKW = ids_vec_reprojEdgeKptWraps.begin(); itREKW != ids_vec_reprojEdgeKptWraps.end(); itREKW++){
        for (auto& reprojEdgeKptWrap : itREKW->second) {
            auto pEdge = reprojEdgeKptWrap._pEdge;
            auto localLm = reprojEdgeKptWrap._pLm;
            if (localLm->IsToDelete()) {
                continue;
            }

            TDO_LOG_VERBOSE_FORMAT("Edge2 between keyframe(%d), landmark(%d), chi2 %f, depthPosi %s",
                                        reprojEdgeKptWrap._pShot->_keyFrameID
                                        % reprojEdgeKptWrap._pLm->_landmarkID
                                        % pEdge->chi2()
                                        % std::to_string(reprojEdgeKptWrap.IsDepthPositive()));
            if (chi_sq_3D < pEdge->chi2() || !reprojEdgeKptWrap.IsDepthPositive()) {
                outlier_observations_lms.push_back(std::make_pair(reprojEdgeKptWrap._pShot, reprojEdgeKptWrap._pLm));
            }
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
                TDO_LOG_DEBUG_FORMAT("outlier pair keyframe(%d), landmark(%d)", pKeyfrm->_keyFrameID % pLm->_landmarkID);
                pKeyfrm->DeleteOneObservedLandmark(pLm);
                pLm->DeleteObservation(pKeyfrm);
            }
        }

        for (auto id_localKeyfrm : ids_localKeyframes) {
            auto pLocalKeyfrm = id_localKeyfrm.second;
            auto pKeyfrmVtx = ids_keyfrmVtx[pLocalKeyfrm->_keyFrameID];
            TDO_LOG_DEBUG_FORMAT("updating pose of keyframe(%d)", pLocalKeyfrm->_keyFrameID);
            Mat44_t newKeyframePoseInWorld = pKeyfrmVtx->estimate().to_homogeneous_matrix().inverse().cast<float>();
            newKeyframePoseInWorld = mathutils::FixRxRz<float>(newKeyframePoseInWorld);
            pLocalKeyfrm->SetKeyframePoseInWorld(newKeyframePoseInWorld);
        }

        for (auto id_localLm : ids_localLandmarks) {
            auto pLocalLm = id_localLm.second;
            auto pLmVtx = ids_landmarkPointVtx[pLocalLm->_landmarkID];
            Mat44_t landmarkPoseInWorld = pLocalLm->GetLandmarkPoseInWorld();
            landmarkPoseInWorld.block(0, 3, 3, 1) = pLmVtx->estimate().cast<float>();
            TDO_LOG_DEBUG_FORMAT("updating pose of landmark(%d)", pLocalLm->_landmarkID);
            pLocalLm->SetLandmarkPoseInWorld(landmarkPoseInWorld);
        }

    }

    TDO_LOG_CRITICAL_FORMAT("localBA finished with %d keyfrm, %d landmarks updated. %d outlierPairs!", 
                                    ids_localKeyframes.size() %
                                    ids_localLandmarks.size() %
                                    outlier_observations_lms.size());

}

}  // end of namespace eventobjectslam