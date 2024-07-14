#include "object.h"
#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>
#include <filesystem>

#include <logging.h>
TDO_LOGGER("eventobjectslam.object")


namespace eventobjectslam {

Eigen::MatrixXf GetVerticesOf3DBoundingCylinderForObject(
    const int numVerticesOneSide,
    const float horizontalSize,
    const Vec3_t objectCenterInRefFrame,
    const Vec3_t topCenterPtInRefFrame
){
    Eigen::MatrixXf vertices3DInRefFrame = Eigen::MatrixXf::Zero(3, numVerticesOneSide * 2);
    Vec3_t vOc2TopC = topCenterPtInRefFrame - objectCenterInRefFrame;
    vOc2TopC /= vOc2TopC.norm();
    Vec3_t vOc2Cam = -objectCenterInRefFrame;
    vOc2Cam /= vOc2Cam.norm();
    Vec3_t vRadius = vOc2TopC.cross(vOc2Cam);
    vRadius /= vRadius.norm();
    Vec3_t vRotate = vRadius;
    float theta = M_PI / 2;
    for (size_t indexVertex = 0; indexVertex < numVerticesOneSide; indexVertex++){
        // rodrigues' formula
        vRotate = vRotate * std::cos(theta) + (vOc2TopC.cross(vRotate)) * std::sin(theta) + vOc2TopC * (vOc2TopC.dot(vRotate)) * (1 - std::cos(theta));
        Vec3_t oneVertex = topCenterPtInRefFrame + vRotate * horizontalSize / 2;
        vertices3DInRefFrame.col(indexVertex) = oneVertex;
    }
    vRadius = vOc2TopC.cross(vOc2Cam);
    vRadius /= vRadius.norm();
    theta = -theta;  // Note: keep the connecting edges between two sides straight.
    for (size_t indexVertex = numVerticesOneSide; indexVertex < numVerticesOneSide * 2; indexVertex++){
        // rodrigues' formula
        vRotate = vRotate * std::cos(theta) + (vOc2TopC.cross(vRotate)) * std::sin(theta) + vOc2TopC * (vOc2TopC.dot(vRotate)) * (1 - std::cos(theta));
        Vec3_t oneVertex = 2 * objectCenterInRefFrame - topCenterPtInRefFrame + vRotate * horizontalSize / 2;
        vertices3DInRefFrame.col(indexVertex) = oneVertex;
    }
    return vertices3DInRefFrame;
}

bool CompareDetectionScoreIfBetter(std::string methodName, float oldScore, float newScore){
    if (methodName == "linemod"){
        return newScore < oldScore;
    }
    else{
        // error, return False for now.
        return false;
    }
}

namespace object{

ObjectBase::ObjectBase(const std::string objectName)
: _objectName(objectName)
{
    TDO_LOG_DEBUG_FORMAT("Object %s initialized without templates!", _objectName);
}

} // end of namespace object

TwoDBoundingBox::TwoDBoundingBox(const float x, const float y, const float bWidth, const float bHeight, const std::shared_ptr<object::ObjectBase> pObjectInfo, const float detectionScore, const std::vector<Vec2_t> keypts)
:_centerX(x), _centerY(y), _bWidth(bWidth), _bHeight(bHeight), _pObjectInfo(pObjectInfo), _detectionScore(detectionScore), _keypts(keypts)
{
    TDO_LOG_DEBUG_FORMAT("created 2d bbox for object %s with %d keypts!",  pObjectInfo->_objectName % _keypts.size());
}

}  // end of eventobjectslam