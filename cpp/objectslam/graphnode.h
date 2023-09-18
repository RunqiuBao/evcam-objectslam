#ifndef EVENTOBJECTSLAM_GRAPHNODE_H
#define EVENTOBJECTSLAM_GRAPHNODE_H

#include <mutex>
#include <map>
#include <vector>

#include "keyframe.h"

namespace eventobjectslam {

class GraphNode {

public:
    explicit GraphNode(KeyFrame* pHostKeyframe);

    ~GraphNode() = default;

    // ============ covisibility graph methods =============
    /*
     *  Compute covisibilities by refering landmark observations.
     *  Execute this method when creating a keyframe.
     */
    void ComputeCovisibility();

    /*
     *  Add connection between this node and another keyframe 
     */
    void AddCovisibilityConnection(std::shared_ptr<KeyFrame> pAnotherKeyframe, const unsigned int weight);

    std::vector<std::shared_ptr<KeyFrame>> GetOrderedCovisibilities() const;

    void UpdateEraseOneCovisibleLandmark(std::map<std::shared_ptr<KeyFrame>, unsigned int> observationsForOneLandmark);

private:
    /*
     *  After editing connection, need to update covisibilities orders by weights.
     */
    void _UpdateCovisibilityOrders();

    std::shared_ptr<KeyFrame> _pHostKeyframe;

    // all covisible keyframes and their weights to hostKeyframe. Weights means the number of covisible landmarks.
    std::map<std::shared_ptr<KeyFrame>, unsigned int> _covisibilityKeyframes_and_weights;

    // minumum thresholds for covisibility graph connection.
    static constexpr unsigned int _weightThreshold = 4;

    // covisibility keyframes in descending order of weights.
    std::vector<std::shared_ptr<KeyFrame>> _orderedCovisibilities;

    // mutable std::mutex _mtxCovisibilityConnections;  // !Note: make sure no race condition in keyframe class side.


};

}  // end of namespace eventobjectslam

#endif  // EVENTOBJECTSLAM_GRAPHNODE_H