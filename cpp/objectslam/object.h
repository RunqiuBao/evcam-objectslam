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
    std::shared_ptr<object::ObjectBase> _pObjectInfo;

    TwoDBoundingBox(const float x, const float y, const float bWidth, const float bHeight, const int templateID, const float templateScale, const std::shared_ptr<object::ObjectBase> pObjectInfo)
    :_centerX(x), _centerY(y), _bWidth(bWidth), _bHeight(bHeight), _templateID(templateID), _templateScale(templateScale), _pObjectInfo(pObjectInfo)
    {}

    void Set3DDepth(const float distance){_esitmated3DDepth = distance;}

};

class ThreeDDetection {

public:
    unsigned int _cameraID;
    Mat44_t _objectInCameraTransform;
    int _detectionID;
    Eigen::MatrixXf _vertices3DInCamera; // Note: 3x8 matrix, vertex 1, 2, 3, 4 should respectively have a connecting edge with 5, 6, 7, 8.

    ThreeDDetection(const Mat44_t objectInCameraTransform, const int detectionID, const Eigen::MatrixXf& vertices3DInCamera, const unsigned int cameraID)
    : _objectInCameraTransform(objectInCameraTransform), _detectionID(detectionID), _vertices3DInCamera(vertices3DInCamera), _cameraID(cameraID)
    {}

};

typedef std::array<float, 6> ThreeDPlane;  // {x, y, z, nx, ny, nz}

}  // end of namespace eventobjectslam

#endif