#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>


namespace tooldetectobject{

class EventLineModTemplate{

public:
    EventLineModTemplate(const cv::Mat image, const Eigen::Matrix4f simulationCamInObjectTransform, const int templateId, float resize=0.33, int intensityThreshold=10);

    GetFeatureVector();

private:
    cv::Mat _image;
    Eigen::Matrix4f _simulationCamInObjectTransform;
    int _templateId;
    float _templateSparsity;
    Eigen::MatrixXi _featureVector;
    Eigen::MatrixXi _featurePointsX;
    Eigen::MatrixXi _featurePointsY;
    int _imageW;
    int _imageH;
    float _scaleFactorCache;

};


class EventLineModTemplateManager{

public:
    EventLineModTemplateManager(){}

private:
    std::vector<EventLineModTemplate> _templateList;
    std::vector<int> _templateIdList;
    float _scale;

    
};

};  // end of namespace tooldetect