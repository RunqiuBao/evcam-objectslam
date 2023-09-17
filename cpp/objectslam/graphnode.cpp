#include "graphnode.h"

#include <logging.h>
TDO_LOGGER("eventobjectslam.graphnode")


namespace eventobjectslam {

GraphNode::GraphNode(KeyFrame* pHostKeyframe) {
    std::shared_ptr<KeyFrame> pHostKeyframeShared(pHostKeyframe);
    _pHostKeyframe = pHostKeyframeShared;
}

void GraphNode::UpdateCovisibilityOrders() {
    std::lock_guard<std::mutex> lock(_mtxCovisibilityConnections);

    std::vector<std::pair<unsigned int, std::shared_ptr<KeyFrame>>> weight_keyframe_pairs;
    weight_keyframe_pairs.reserve(_covisibilityKeyframes_and_weights.size());

    for (const auto& keyframe_weight : _covisibilityKeyframes_and_weights) {
        weight_keyframe_pairs.push_back(std::make_pair(keyframe_weight.second, keyframe_weight.first));
    }

    // sort keyframes by weights.
    // Note: sorting will use the first Byte to sort.
    std::sort(weight_keyframe_pairs.rbegin(), weight_keyframe_pairs.rend());  // Note: rbegin is an iterator starting from backward.

    _orderedCovisibilities.clear();
    _orderedCovisibilities.reserve(weight_keyframe_pairs.size());
    for (const auto& weight_keyframe_pair : weight_keyframe_pairs) {
        _orderedCovisibilities.push_back(weight_keyframe_pair.second);
    }
}

void GraphNode::AddCovisibilityConnection(std::shared_ptr<KeyFrame> pAnotherKeyframe, const unsigned int weight){
    bool bNeedUpdateOrder = false;
    {
        // scope of mutex lock
        std::lock_guard<std::mutex> lock(_mtxCovisibilityConnections);
        if (!_covisibilityKeyframes_and_weights.count(pAnotherKeyframe)) {
            // if this keyframe did not exist.
            _covisibilityKeyframes_and_weights[pAnotherKeyframe] = weight;
            bNeedUpdateOrder = true;
        }
        else if (_covisibilityKeyframes_and_weights.at(pAnotherKeyframe) != weight) {
            // if the weight is updated.
            _covisibilityKeyframes_and_weights.at(pAnotherKeyframe) = weight;
            bNeedUpdateOrder = true;
        }
    }

    if (bNeedUpdateOrder) {
        UpdateCovisibilityOrders();
    }
}

void GraphNode::ComputeCovisibility() {
    const auto landmarks = _pHostKeyframe->GetLandmarks();
    std::map<std::shared_ptr<KeyFrame>, unsigned int> keyframes_weights;
    for (const auto landmark : landmarks) {
        const auto observations = landmark->GetObservations();
        for (const auto& obs : observations) {
            auto oneKeyframe = obs.first;
            if (*oneKeyframe == *_pHostKeyframe) {
                continue;
            }
            // count weight for the keyframe.
            keyframes_weights[oneKeyframe]++;
        }
    }

    if (keyframes_weights.empty()){
        return;
    }

    unsigned int maxWeight = 0;
    std::shared_ptr<KeyFrame> pNearestCovisiblilityKeyframe = nullptr;

    std::vector<std::pair<unsigned int, std::shared_ptr<KeyFrame>>> vector_weight_covisibility;
    vector_weight_covisibility.reserve(keyframes_weights.size());
    for (const auto& keyframe_weight : keyframes_weights) {
        auto pCovisibilityKeyframe = keyframe_weight.first;
        const auto weight = keyframe_weight.second;

        if (weight >= maxWeight) {
            maxWeight = weight;
            pNearestCovisiblilityKeyframe = pCovisibilityKeyframe;
        }

        if (weight > _weightThreshold) {
            vector_weight_covisibility.push_back(std::make_pair(weight, pCovisibilityKeyframe));
        }
    }
    // add the maximum one if empty
    if (vector_weight_covisibility.empty()) {
        vector_weight_covisibility.push_back(std::make_pair(maxWeight, pNearestCovisiblilityKeyframe));
    }

    // add this keyframe to each covisibility keyframe's covisibility.
    for (const auto& weight_covisibility : vector_weight_covisibility) {
        auto pCovisibilityKeyframe = weight_covisibility.second;
        const auto weight = weight_covisibility.first;
        pCovisibilityKeyframe->_graphNode->AddCovisibilityConnection(_pHostKeyframe, weight);
    }

    // sort by weights
    std::sort(vector_weight_covisibility.rbegin(), vector_weight_covisibility.rend());  // Note: rbegin is backward iterator. it becomes descending order sort thus.
    
    decltype(_orderedCovisibilities) orderedCovisibilities;
    orderedCovisibilities.reserve(vector_weight_covisibility.size());
    for (const auto& weight_covisibility : vector_weight_covisibility) {
        orderedCovisibilities.push_back(weight_covisibility.second);
    }

    {
        // scope of mutex
        std::lock_guard<std::mutex> lock(_mtxCovisibilityConnections);

        _covisibilityKeyframes_and_weights = keyframes_weights;
        _orderedCovisibilities = orderedCovisibilities;
    }
    TDO_LOG_INFO_FORMAT("succeeded ComputeCovisibility for keyfrm: %d", _pHostKeyframe->_keyFrameID);
}

}  // end of namespace eventobjectslam