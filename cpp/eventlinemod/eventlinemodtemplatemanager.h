#pragma once

#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <filesystem>
#include <stdexcept>

#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <Eigen/Core>

#include "Quadtree.h"


namespace tooldetectobject{


struct Valid2DPoint{
    quadtree::Box<float> zeroSizeBox;
    std::size_t id;
};

class TemplateInfo{

public:
    TemplateInfo(const size_t templId, const Eigen::Matrix4f camInObjectTransformation)
    :_templId(templId), _camInObjectTransformation(camInObjectTransformation)
    {}

    size_t _templId;
    Eigen::Matrix4f _camInObjectTransformation;

};


class EventLineModTemplate{

public:
    EventLineModTemplate(
        const cv::Mat image,
        const Eigen::Matrix4f simulationCamInObjectTransform,
        const int templateId,
        float resize=0.33,
        int intensityThreshold=10
    );

    static void GetFeatureVector(
        const cv::Mat& image,
        Eigen::MatrixXi& featurePointsX,
        Eigen::MatrixXi& featurePointsY,
        Eigen::MatrixXi& featureVector,
        float& templateSparsity,
        const int numFeaturePoints=64,
        const int maxNumPointsQuadtreeNode=10,
        const float gradMagnitudeThreshold=100.0,
        const float isNoiseThreshold=40.0
    );

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
    EventLineModTemplateManager(const std::string templatePath);

private:
    std::vector<EventLineModTemplate> _templateList;
    std::vector<int> _templateIdList;
    // float _scale;

    
};

};  // end of namespace tooldetect