#ifndef EVENTOBJECTSLAM_OBJECT_H
#define EVENTOBJECTSLAM_OBJECT_H

#include "objectslam.h"

namespace eventobjectslam{

typedef std::array<float, 6> ThreeDPlane;  // {x, y, z, nx, ny, nz}


/***** shared methods operating objects *****/
// inline Eigen::MatrixXf GetVerticesOf3DBoundingBoxFromObject(const std::shared_ptr<object::ObjectBase> pObjectInfo){
//     float xSize, ySize, zSize;
//     xSize = pObjectInfo->_objectExtents[0];
//     ySize = pObjectInfo->_objectExtents[1];
//     zSize = pObjectInfo->_objectExtents[2];
//     Eigen::MatrixXf vertices3D;
//     vertices3D.resize(3, 8);
//     vertices3D.col(0) << xSize / 2, ySize / 2, zSize / 2;
//     vertices3D.col(1) << -xSize / 2, ySize / 2, zSize / 2;
//     vertices3D.col(2) << -xSize / 2, -ySize / 2, zSize / 2;
//     vertices3D.col(3) << xSize / 2, -ySize / 2, zSize / 2;
//     vertices3D.col(4) << xSize / 2, ySize / 2, -zSize / 2;
//     vertices3D.col(5) << -xSize / 2, ySize / 2, -zSize / 2;
//     vertices3D.col(6) << -xSize / 2, -ySize / 2, -zSize / 2;
//     vertices3D.col(7) << xSize / 2, -ySize / 2, -zSize / 2;
//     return vertices3D;
// }


Eigen::MatrixXf GetVerticesOf3DBoundingCylinderForObject(
    const int numVerticesOneSide,
    const float horizontalSize,
    const Vec3_t objectCenterInRefFrame,
    const Vec3_t topCenterPtInRefFrame
);

bool CompareDetectionScoreIfBetter(std::string methodName, float oldScore, float newScore);

namespace object{

// deprecating usage of this class.
class ObjectBase {

public:
    std::string _objectName;

    ObjectBase(const std::string objectName);

};

}  // end of namespace object


class TwoDBoundingBox {

public:
    float _centerX;
    float _centerY;
    float _bWidth;
    float _bHeight;
    float _esitmated3DDepth;
    float _detectionScore;
    std::vector<Vec2_t> _keypts;
    std::shared_ptr<object::ObjectBase> _pObjectInfo;

    TwoDBoundingBox(const float x, const float y, const float bWidth, const float bHeight, const std::shared_ptr<object::ObjectBase> pObjectInfo, const float detectionScore, const std::vector<Vec2_t> keypts);

    void Set3DDepth(const float distance){_esitmated3DDepth = distance;}

};

class ThreeDDetection {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    unsigned int _cameraID;
    Vec3_t _objectCenterInRefFrame;  // Note: use center->keypt vector to express orientation. refFrame is the frame that observed this detection.
    Vec3_t _keypt1InRefFrame;
    Eigen::MatrixXf _vertices3DInRefFrame; // Note: 3x8 matrix, vertex 1, 2, 3, 4 should respectively have a connecting edge with 5,6,7,8. Note the vertices form a cylinder and keypt1 is at top center.

    float _horizontalSize;  // diameter of the cylinder.
    int _detectionID;
    float _detectionScore;  // Note: used for records of confidence of the detection.
    std::shared_ptr<TwoDBoundingBox> _pLeftBbox;  // Note; should not be changed after initialization.
    std::shared_ptr<TwoDBoundingBox> _pRightBbox;
    std::shared_ptr<object::ObjectBase> _pObjectInfo;

    ThreeDDetection(
        const Vec3_t objectCenterInRefFrame,
        const int detectionID,
        const unsigned int cameraID,
        const float detectionScore,
        const  std::shared_ptr<TwoDBoundingBox> pLeftBbox,
        const  std::shared_ptr<TwoDBoundingBox> pRightBbox,
        const Vec3_t& keypt1InRefFrame,
        const float horizontalSize,
        const std::shared_ptr<object::ObjectBase> pObjectInfo
    )
    : _objectCenterInRefFrame(objectCenterInRefFrame),
      _detectionID(detectionID),
      _cameraID(cameraID),
      _detectionScore(detectionScore),
      _pLeftBbox(pLeftBbox),
      _pRightBbox(pRightBbox),
      _keypt1InRefFrame(keypt1InRefFrame),
      _horizontalSize(horizontalSize),
      _pObjectInfo(pObjectInfo)
    {
        _vertices3DInRefFrame = GetVerticesOf3DBoundingCylinderForObject(
            4,
            _horizontalSize,
            _objectCenterInRefFrame,
            _keypt1InRefFrame
        );
    }

    ThreeDDetection(){}  // default constructor.

    void initialize(const ThreeDDetection& other){
        _objectCenterInRefFrame = other._objectCenterInRefFrame;
        _detectionID = other._detectionID;
        _cameraID = other._cameraID;
        _detectionScore = other._detectionScore;
        _pLeftBbox = other._pLeftBbox;
        _pRightBbox = other._pRightBbox;
        _keypt1InRefFrame = other._keypt1InRefFrame;
        _horizontalSize = other._horizontalSize;
        _pObjectInfo = other._pObjectInfo;
        _vertices3DInRefFrame = GetVerticesOf3DBoundingCylinderForObject(
            4,
            _horizontalSize,
            _objectCenterInRefFrame,
            _keypt1InRefFrame
        );
    }

};

class RefObject {

public:
    ThreeDDetection _detection;
    int _refObjectIDInKeyframe;

    RefObject(const ThreeDDetection& detection, const int refObjectIDInKeyframe)
    : _detection(detection), _refObjectIDInKeyframe(refObjectIDInKeyframe)
    {}

};

}  // end of namespace eventobjectslam

#endif