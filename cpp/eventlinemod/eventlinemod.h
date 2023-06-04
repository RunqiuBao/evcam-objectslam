#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

#include "eventlinemodtemplatemanager.h"

namespace tooldetectobject{


class BBox{

public:
    BBox(const int topLeftX, const int topLeftY, const int bottomRightX, const int bottomRightY)
    : _topLeftX(topLeftX), _topLeftY(topLeftY), _bottomRightX(bottomRightX), _bottomRightY(bottomRightY)
    {}

private:
    int _topLeftX;
    int _topLeftY;
    int _bottomRightX;
    int _bottomRightY;
};


class EventLineModDetection{

public:
    /** \brief Constructor. */
    EventLineModDetection(const int x, const int y, const int templateIndex, const float score, const float scale, const BBox bbox) 
    : _x(x), _y(y), _templateIndex(templateIndex), _score(score), _scale(scale), _bbox(bbox)
    {}

private:
    /** \brief x-position of the detection. */
    int _x;
    /** \brief y-position of the detection. */
    int _y;
    /** \brief index (ID) of the detected template. */
    int _templateIndex;
    /** \brief score of the detection. */
    float _score;
    /** \brief scale at which the template was detected. */
    float _scale;
    BBox _bbox;

};

class EventLineModDetector{

public:
    /** \brief Constructor */
    EventLineModDetector (const std::string templatePath, const float templateResponseThreshold)
    : _templateManager(EventLineModTemplateManager(templatePath)), _templateResponseThreshold(templateResponseThreshold)
    {}

    /**
     * @brief use linemod style template matching method to detect 2d object bounding box
     * 
     * @param inputFrame
     * @param minScale                     min scale for template.     
     * @param maxScale                     max scale for template.
     * @param scaleMultiplier              factor to multiply when upscaling. newScale = currentScale * scaleMultiplier
     * @param scanStep                     minimum pixels to move during template matching. 
     * @param isTooSparseThreshold         when fired area whthin a patch smaller than this threshold, regard it as too sparse. skip the patch.
     * @param isShow                       for debug.s
     * @return std::vector<EventLineModDetection>
     */
    std::vector<EventLineModDetection> DetectTemplatesSemiScaleInvariant(
        cv::Mat& inputFrame,
        const float minScale = 0.6944,
        const float maxScale = 1.44,
        const float scaleMultiplier = 1.2,
        const int scanStep = 4,
        const bool isTooSparseThreshold = 0.5,
        const bool isShow = false
    );


private:
    EventLineModTemplateManager _templateManager;
    float _templateResponseThreshold;

};

};  // end of namespace tooldetect
