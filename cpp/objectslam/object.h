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
    std::vector<Vec2_t> _facetCorners;  // from topleft corner in clockwise direction.
    std::shared_ptr<object::ObjectBase> _pObjectInfo;

    TwoDBoundingBox(
        const float x,
        const float y,
        const float bWidth,
        const float bHeight,
        const std::shared_ptr<object::ObjectBase> pObjectInfo,
        const float detectionScore,
        const std::vector<Vec2_t> facetCorners
    );

    void Set3DDepth(const float distance){_esitmated3DDepth = distance;}

};

class ThreeDDetection {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    unsigned int _cameraID;
    Vec3_t _objectCenterInRefFrame;  // Note: use center->keypt vector to express orientation. refFrame is the frame that observed this detection.
    std::vector<Vec3_t> _facetCornersInRefFrame;
    Vec3_t _normalOfFacet;
    bool hasFacet;

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
        const std::vector<Vec3_t> facetCornersInRefFrame,
        const std::shared_ptr<object::ObjectBase> pObjectInfo
    )
    : _objectCenterInRefFrame(objectCenterInRefFrame),
      _detectionID(detectionID),
      _cameraID(cameraID),
      _detectionScore(detectionScore),
      _pLeftBbox(pLeftBbox),
      _pRightBbox(pRightBbox),
      _facetCornersInRefFrame(facetCornersInRefFrame),
      _pObjectInfo(pObjectInfo)
    {
        if (_facetCornersInRefFrame.size() > 0){
            Vec3_t c1c4 = facetCornersInRefFrame[3] - facetCornersInRefFrame[0];
            Vec3_t c1c2 = facetCornersInRefFrame[1] - facetCornersInRefFrame[0];

            _normalOfFacet = c1c4.cross(c1c2);
            _normalOfFacet /= _normalOfFacet.norm();
            _horizontalSize = (facetCornersInRefFrame[0] - facetCornersInRefFrame[1]).norm();
            hasFacet = true;
        }
        else{
            hasFacet = false;
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
        _facetCornersInRefFrame = other._facetCornersInRefFrame;
        _horizontalSize = other._horizontalSize;
        _pObjectInfo = other._pObjectInfo;
        _normalOfFacet = other._normalOfFacet;

    }

    const Eigen::MatrixXf GetFacetCornersInEigen(){
        Eigen::MatrixXf mFacetCorners(3, 4);
        for (int indexCorner=0; indexCorner < 4; indexCorner++){
            mFacetCorners.col(indexCorner) = _facetCornersInRefFrame[indexCorner];
        }
        return mFacetCorners;
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