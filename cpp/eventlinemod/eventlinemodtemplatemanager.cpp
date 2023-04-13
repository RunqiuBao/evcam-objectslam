#include "eventlinemodtemplatemanager.h"

#include <logging.h>
TDO_LOGGER("eventlinemod.templatemanager")


tooldetectobject::EventLineModTemplate::EventLineModTemplate(const cv::Mat image, const Eigen::Matrix4f simulationCamInObjectTransform, const int templateId, float resize, int intensityThreshold)
: _simulationCamInObjectTransform(simulationCamInObjectTransform), _templateId(templateId)
{
    _simulationCamInObjectTransform.block(0, 3, 3, 1) /= resize;
    cv::resize(image, _image, cv::Size(), resize, resize);
    _imageW = _image.col();
    _imageH = _image.row();
    cv::Mat templateNoiseMask;
    cv::threshold(_image, templateNoiseMask, 0, intensityThreshold, cv::THRESH_BINARY);
    _image.setTo(cv::Scalar(0), templateNoiseMask);
    GetFeatureVector();
    _scaleFactorCache = resize;

}

tooldetectobject::EventLineModTemplate::GetFeatureVector(){
    
}