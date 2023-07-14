#ifndef EVENTOBJECTSLAM_OBJECT_H
#define EVENTOBJECTSLAM_OBJECT_H

#include "objectslam.h"

namespace eventobjectslam{

namespace object{

class ObjectTemplate {

public:
    uint16_t _templID;
    Mat44_t _simulationCameraInObjectTransform;

    ObjectTemplate(const uint16_t templID, const Mat44_t simulationCameraInObjectTransform)
    :_templID(templID), _simulationCameraInObjectTransform(simulationCameraInObjectTransform)
    {}

};

class ObjectBase {

public:
    std::string _objectName;
    ObjectExtents _objectExtents;
    std::vector<ObjectTemplate> _templates;

    ObjectBase(const std::string sTemplateInfoPath);

};

}  // end of namespace object

}  // end of namespace eventobjectslam

#endif