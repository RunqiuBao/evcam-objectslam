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
    std::vector<size_t> indicesInTemplatesArray;  // Note: input templateID, output index in _templates.

    ObjectBase(const std::string sTemplatesPath);

};

}  // end of namespace object

}  // end of namespace eventobjectslam

#endif