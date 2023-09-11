#ifndef EVENTOBJECTSLAM_OBJECT_H
#define EVENTOBJECTSLAM_OBJECT_H

#include "objectslam.h"

namespace eventobjectslam{

namespace object{

class ObjectTemplate {

public:
    size_t _templID;
    Mat44_t _simulationCameraInObjectTransform;

    ObjectTemplate(const size_t templID, const Mat44_t simulationCameraInObjectTransform)
    :_templID(templID), _simulationCameraInObjectTransform(simulationCameraInObjectTransform)
    {}

};

class ObjectBase {

public:
    std::string _objectName;
    ObjectExtents _objectExtents;
    std::vector<ObjectTemplate> _templates;
    std::vector<size_t> _indicesInTemplatesArray;  // Note: input templateID, output index in _templates.

    ObjectBase(const std::string sTemplatesPath);

};

}  // end of namespace object


class TwoDBoundingBox {

public:
    float _centerX;
    float _centerY;
    float _bWidth;
    float _bHeight;
    int _templateID;
    float _templateScale;
    float _esitmated3DDepth;
    float _detectionScore;
    std::shared_ptr<object::ObjectBase> _pObjectInfo;

    TwoDBoundingBox(const float x, const float y, const float bWidth, const float bHeight, const int templateID, const float templateScale, const std::shared_ptr<object::ObjectBase> pObjectInfo, const float detectionScore)
    :_centerX(x), _centerY(y), _bWidth(bWidth), _bHeight(bHeight), _templateID(templateID), _templateScale(templateScale), _pObjectInfo(pObjectInfo), _detectionScore(detectionScore)
    {}

    void Set3DDepth(const float distance){_esitmated3DDepth = distance;}

};

class ThreeDDetection {

public:
    unsigned int _cameraID;
    Mat44_t _objectInCameraTransform;
    int _detectionID;
    float _detectionScore;  // Note: used for records of confidence of the detection.
    Eigen::MatrixXf _vertices3DInCamera; // Note: 3x8 matrix, vertex 1, 2, 3, 4 should respectively have a connecting edge with 5, 6, 7, 8.

    ThreeDDetection(const Mat44_t objectInCameraTransform, const int detectionID, const Eigen::MatrixXf& vertices3DInCamera, const unsigned int cameraID, const float detectionScore)
    : _objectInCameraTransform(objectInCameraTransform), _detectionID(detectionID), _vertices3DInCamera(vertices3DInCamera), _cameraID(cameraID), _detectionScore(detectionScore)
    {}

};

class RefObject {

public:
    ThreeDDetection _detection;
    int _refObjectIDInKeyframe;

    RefObject(const ThreeDDetection& detection, const int refObjectIDInKeyframe)
    : _detection(detection), _refObjectIDInKeyframe(refObjectIDInKeyframe)
    {}

};

typedef std::array<float, 6> ThreeDPlane;  // {x, y, z, nx, ny, nz}


/***** shared methods operating objects *****/
inline Eigen::MatrixXf GetVerticesOf3DBoundingBoxFromObject(const std::shared_ptr<object::ObjectBase> pObjectInfo){
    float xSize, ySize, zSize;
    xSize = pObjectInfo->_objectExtents[0];
    ySize = pObjectInfo->_objectExtents[1];
    zSize = pObjectInfo->_objectExtents[2];
    Eigen::MatrixXf vertices3D;
    vertices3D.resize(3, 8);
    vertices3D.col(0) << xSize / 2, ySize / 2, zSize / 2;
    vertices3D.col(1) << -xSize / 2, ySize / 2, zSize / 2;
    vertices3D.col(2) << -xSize / 2, -ySize / 2, zSize / 2;
    vertices3D.col(3) << xSize / 2, -ySize / 2, zSize / 2;
    vertices3D.col(4) << xSize / 2, ySize / 2, -zSize / 2;
    vertices3D.col(5) << -xSize / 2, ySize / 2, -zSize / 2;
    vertices3D.col(6) << -xSize / 2, -ySize / 2, -zSize / 2;
    vertices3D.col(7) << xSize / 2, -ySize / 2, -zSize / 2;
    return vertices3D;
}

inline bool CompareDetectionScoreIfBetter(std::string methodName, float oldScore, float newScore){
    if (methodName == "linemod"){
        return newScore < oldScore;
    }
    else{
        // error, return False for now.
        return false;
    }
}

}  // end of namespace eventobjectslam

#endif