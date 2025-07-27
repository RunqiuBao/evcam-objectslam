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
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    float _centerX;
    float _centerY;
    float _bWidth;
    float _bHeight;
    float _esitmated3DDepth;
    float _detectionScore;
    bool _hasFacet;
    std::vector<Vec2_t> _keypts;
    std::vector<Vec2_t> _vertices2D;  // from topleft corner in clockwise direction.
    std::shared_ptr<object::ObjectBase> _pObjectInfo;

    TwoDBoundingBox(
        const float x,
        const float y,
        const float bWidth,
        const float bHeight,
        const std::shared_ptr<object::ObjectBase> pObjectInfo,
        const float detectionScore,
        const std::vector<Vec2_t> vertices2D,
        const std::vector<Vec2_t> keypts,
        const bool hasFacet
    );

    void Set3DDepth(const float distance){_esitmated3DDepth = distance;}

};

class ThreeDDetection {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    unsigned int _cameraID;
    Vec3_t _objectCenterInRefFrame;  // Note: use center->keypt vector to express orientation. refFrame is the frame that observed this detection.
    std::vector<Vec3_t> _vertices3DInRefFrame;
    Vec3_t _normalOfFacet;
    bool _hasFacet;

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
        const std::vector<Vec3_t> vertices3DInRefFrame,
        const std::shared_ptr<object::ObjectBase> pObjectInfo,
        const float horizontalSize
    )
    : _objectCenterInRefFrame(objectCenterInRefFrame),
      _detectionID(detectionID),
      _cameraID(cameraID),
      _detectionScore(detectionScore),
      _pLeftBbox(pLeftBbox),
      _pRightBbox(pRightBbox),
      _vertices3DInRefFrame(vertices3DInRefFrame),  // is not hasFacet, vertices3DInRefFrame has two keypoints, {top, center}
      _pObjectInfo(pObjectInfo),
      _horizontalSize(horizontalSize)
    {
        if (pLeftBbox->_hasFacet && pRightBbox->_hasFacet){
            assert(_vertices3DInRefFrame.size() >= 3);
            Vec3_t c1c3 = _vertices3DInRefFrame[2] - _vertices3DInRefFrame[0];
            Vec3_t c1c2 = _vertices3DInRefFrame[1] - _vertices3DInRefFrame[0];

            _normalOfFacet = c1c3.cross(c1c2);
            _normalOfFacet /= _normalOfFacet.norm();
            _hasFacet = true;
        }
        else{
            // vertices3DInRefFrame has two keypoints, {top, center}
            assert(_vertices3DInRefFrame.size() == 2);
            int numBoundingCylinderVerticesHalfSide = 4;
            Eigen::MatrixXf boundingVertices3DInLandmark = GetVerticesOf3DBoundingCylinderForObject(numBoundingCylinderVerticesHalfSide, horizontalSize, vertices3DInRefFrame[1], vertices3DInRefFrame[0]);
            // add bounding vertices to _vertices3DInRefFrame
            for (int indexVertex = 0; indexVertex < numBoundingCylinderVerticesHalfSide * 2; indexVertex++){
                _vertices3DInRefFrame.push_back(boundingVertices3DInLandmark.col(indexVertex));
            }
            _hasFacet = false;
        }

    }

    ThreeDDetection(){}  // default constructor.

    void assign(const ThreeDDetection& other){
        _objectCenterInRefFrame = other._objectCenterInRefFrame;
        _detectionID = other._detectionID;
        _cameraID = other._cameraID;
        _detectionScore = other._detectionScore;
        _pLeftBbox = other._pLeftBbox;
        _pRightBbox = other._pRightBbox;
        _vertices3DInRefFrame = other._vertices3DInRefFrame;
        _horizontalSize = other._horizontalSize;
        _pObjectInfo = other._pObjectInfo;
        _hasFacet = other._hasFacet;
        if (!_hasFacet){
            int numBoundingCylinderVerticesHalfSide = 4;
            Eigen::MatrixXf boundingVertices3DInLandmark = GetVerticesOf3DBoundingCylinderForObject(numBoundingCylinderVerticesHalfSide, _horizontalSize, _vertices3DInRefFrame[1], _vertices3DInRefFrame[0]);
            // add bounding vertices to _vertices3DInRefFrame
            for (int indexVertex = 0; indexVertex < numBoundingCylinderVerticesHalfSide * 2; indexVertex++){
                _vertices3DInRefFrame.push_back(boundingVertices3DInLandmark.col(indexVertex));
            }
        }
        else{
            _normalOfFacet = other._normalOfFacet;
        }
    }

    const Eigen::MatrixXf GetVertices3DInEigen(){
        Eigen::MatrixXf mVertices3D(3, _vertices3DInRefFrame.size());
        for (int indexCorner=0; indexCorner < _vertices3DInRefFrame.size(); indexCorner++){
            mVertices3D.col(indexCorner) = _vertices3DInRefFrame[indexCorner];
        }
        return mVertices3D;
    }

};

class RefObject {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ThreeDDetection _detection;
    int _refObjectIDInKeyframe;

    RefObject(const ThreeDDetection& detection, const int refObjectIDInKeyframe)
    : _detection(detection), _refObjectIDInKeyframe(refObjectIDInKeyframe)
    {}

};

}  // end of namespace eventobjectslam

#endif