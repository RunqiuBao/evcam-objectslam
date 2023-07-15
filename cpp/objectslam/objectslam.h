#ifndef EVENTOBJECTSLAM_H
#define EVENTOBJECTSLAM_H

#include <Eigen/Core>
#include <memory>

namespace eventobjectslam {

typedef Eigen::Matrix4f Mat44_t;
typedef std::array<float, 3> ObjectExtents;

class TwoDBoundingBox {

public:
    float _centerX;
    float _centerY;
    float _bWidth;
    float _bHeight;
    int _templateID;
    float _templateScale;
    float _esitmated3DDistance;

    TwoDBoundingBox(const float x, const float y, const float bWidth, const float bHeight, const int templateID, const float templateScale)
    :_centerX(x), _centerY(y), _bWidth(bWidth), _bHeight(bHeight), _templateID(templateID), _templateScale(templateScale)
    {}

    void Set3DDistance(const float distance){_esitmated3DDistance = distance;}

};

class ThreeDDetection {

public:
    unsigned int _cameraID;
    Mat44_t _objectInCameraTransform;
    int _detectionID;
    Eigen::MatrixXf _vertices3DInCamera; // Note: 8x3 matrix, vertex 1, 2, 3, 4 should respectively have a connecting edge with 5, 6, 7, 8.

    ThreeDDetection(const Mat44_t objectInCameraTransform, const int detectionID, const Eigen::MatrixXf& vertices3DInCamera, const unsigned int cameraID)
    : _objectInCameraTransform(objectInCameraTransform), _detectionID(detectionID), _vertices3DInCamera(vertices3DInCamera), _cameraID(cameraID)
    {}

};

}

#endif  // EVENTOBJECTSLAM_H