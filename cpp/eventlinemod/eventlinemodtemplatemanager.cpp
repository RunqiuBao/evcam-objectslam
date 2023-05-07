#include "eventlinemodtemplatemanager.h"


#include <logging.h>
TDO_LOGGER("eventlinemod.templatemanager")


static tooldetectobject::Valid2DPoint GenerateValid2DPointFromCoordinates(const float xCoord, const float yCoord, const size_t pointId){
    tooldetectobject::Valid2DPoint newPoint;
    newPoint.zeroSizeBox.left = xCoord;
    newPoint.zeroSizeBox.top = yCoord;
    newPoint.zeroSizeBox.width = 0.0f;
    newPoint.zeroSizeBox.height = 0.0f;
    newPoint.id = pointId;
    return newPoint;
}

static int _ComputeQuantizedGradientOrientation(
    const cv::Mat imagePatch,
    const int gradMagnitudeThreshold=100,
    const int numSector=8
){
    cv::Mat gradX, gradY, grad;
    cv::Rect coreGradientRoi(1, 1, 3, 3);
    cv::Sobel(imagePatch, gradX, CV_32F, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
    cv::Sobel(imagePatch, gradY, CV_32F, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
    cv::divide(gradY(coreGradientRoi), gradX(coreGradientRoi), grad);
    // nested for loop
    cv::Mat response = cv::Mat::zeros(grad.size(), grad.type());
    cv::Mat binCount = cv::Mat::zeros(cv::Size(grad.cols * grad.rows, 1), CV_8U);
    for (size_t indexX=0; indexX<3; indexX++){
        for (size_t indexY=0; indexY<3; indexY++){
            if (gradX.at<float>(indexY, indexX) < gradMagnitudeThreshold){
                response.at<float>(indexY, indexX) = 0;
                binCount.at<uint8_t>(0, 0) ++;
            }
            else if (grad.at<float>(indexY, indexX) > 0 && grad.at<float>(indexY, indexX) <= tan(M_PI / numSector)){
                response.at<float>(indexY, indexX) = 1;
                binCount.at<uint8_t>(1, 0) ++;
            }
            else if (grad.at<float>(indexY, indexX) > tan(M_PI / numSector) && grad.at<float>(indexY, indexX) <= tan(M_PI * 2 / numSector)){
                response.at<float>(indexY, indexX) = 2;
                binCount.at<uint8_t>(2, 0) ++;
            }
            else if (grad.at<float>(indexY, indexX) > tan(M_PI * 2 / numSector) && grad.at<float>(indexY, indexX) <= tan(M_PI * 3 / numSector)){
                response.at<float>(indexY, indexX) = 3;
                binCount.at<uint8_t>(3, 0) ++;
            }
            else if (grad.at<float>(indexY, indexX) > tan(M_PI * 3 / numSector)){
                response.at<float>(indexY, indexX) = 4;
                binCount.at<uint8_t>(4, 0) ++;
            }
            else if (grad.at<float>(indexY, indexX) <= tan(-1 * M_PI * 3 / numSector)){
                response.at<float>(indexY, indexX) = 5;
                binCount.at<uint8_t>(5, 0) ++;
            }
            else if (grad.at<float>(indexY, indexX) > tan(-1 * M_PI * 3 / numSector) && grad.at<float>(indexY, indexX) <= tan(-1 * M_PI * 2 / numSector)){
                response.at<float>(indexY, indexX) = 6;
                binCount.at<uint8_t>(6, 0) ++;
            }
            else if (grad.at<float>(indexY, indexX) > tan(-1 * M_PI * 2 / numSector) && grad.at<float>(indexY, indexX) <= tan(-1 * M_PI / numSector)){
                response.at<float>(indexY, indexX) = 7;
                binCount.at<uint8_t>(7, 0) ++;
            }
            else if (grad.at<float>(indexY, indexX) > tan(-1 * M_PI / numSector) && grad.at<float>(indexY, indexX) <= 0){
                response.at<float>(indexY, indexX) = 8;
                binCount.at<uint8_t>(8, 0) ++;
            }
            else{
                continue;
            }
        }
    }
    double maxCount;
    cv::Point maxLoc;
    cv::minMaxLoc(binCount, nullptr, &maxCount, nullptr, &maxLoc);
    TDO_LOG_VERBOSE_FORMAT("In this localPatch, mainBin is %d with %d counts.", maxLoc.y % maxCount);
    return maxLoc.y;
}

tooldetectobject::EventLineModTemplate::EventLineModTemplate(
    const cv::Mat image,
    const Eigen::Matrix4f simulationCamInObjectTransform,
    const int templateId,
    float resize,
    int intensityThreshold
)
: _simulationCamInObjectTransform(simulationCamInObjectTransform), _templateId(templateId)
{
    _simulationCamInObjectTransform.block(0, 3, 3, 1) /= resize;
    cv::resize(image, _image, cv::Size(), resize, resize);
    _imageW = _image.cols;
    _imageH = _image.rows;
    cv::Mat templateNoiseMask;
    cv::threshold(_image, templateNoiseMask, 0, intensityThreshold, cv::THRESH_BINARY);
    _image.setTo(cv::Scalar(0), templateNoiseMask);
    GetFeatureVector(
        _image,
        _featurePointsX,
        _featurePointsY,
        _featureVector,
        _templateSparsity
    );
    _scaleFactorCache = resize;

}

/**
 * @brief Randomized and distributed N feature points forming as a featured vector
 * 
 * @param image 
 * @param numFeaturePoints 
 * @param maxNumPointsQuadtreeNode 
 * @param gradMagnitudeThreshold 
 * @param isNoiseThreshold 
 */
void tooldetectobject::EventLineModTemplate::GetFeatureVector(
    const cv::Mat& image,
    Eigen::MatrixXi& featurePointsX,
    Eigen::MatrixXi& featurePointsY,
    Eigen::MatrixXi& featureVector,
    float& templateSparsity,
    const int numFeaturePoints,
    int maxNumPointsQuadtreeNode,
    const float gradMagnitudeThreshold,
    const float isNoiseThreshold
){
    cv::Mat imageLaplacian;
    cv::Laplacian(image, imageLaplacian, CV_32F, 3);
    cv::Mat imageLaplacianNoiseMask;  // Note: same type as imageLaplacian
    cv::threshold(imageLaplacian, imageLaplacianNoiseMask, isNoiseThreshold, 255, cv::THRESH_BINARY);
    imageLaplacianNoiseMask.convertTo(imageLaplacianNoiseMask, CV_8U);
    int morphElem = 0;  // rectangle
    int morphSize = 1;
    cv::Mat element = cv::getStructuringElement(morphElem, cv::Size(2 * morphSize + 1, 2 * morphSize + 1), cv::Point(morphSize, morphSize));
    cv::morphologyEx(imageLaplacianNoiseMask, imageLaplacianNoiseMask, cv::MORPH_CLOSE, element);
    templateSparsity = cv::sum(imageLaplacianNoiseMask)[0] / (imageLaplacianNoiseMask.cols * imageLaplacianNoiseMask.rows * 255);
    // randomly select n feature points
    std::vector<cv::Point> coordinateMaskedPoints;
    cv::findNonZero(imageLaplacianNoiseMask, coordinateMaskedPoints);
    // // adjust numFeaturePoints
    if (coordinateMaskedPoints.size() < (numFeaturePoints * maxNumPointsQuadtreeNode)){
        int marginFactor = 2;
        maxNumPointsQuadtreeNode = coordinateMaskedPoints.size() / numFeaturePoints / marginFactor;
        if (maxNumPointsQuadtreeNode == 0){
            throw std::runtime_error(std::string("template too small and could not extract enough feature points."));
        }
    }
    std::vector<size_t> indicesMaskedPointsRandom(coordinateMaskedPoints.size());
    std::iota(indicesMaskedPointsRandom.begin(), indicesMaskedPointsRandom.end(), 0);
    std::random_device randomDevice;
    std::default_random_engine randomEngine{randomDevice()};
    std::shuffle(indicesMaskedPointsRandom.begin(), indicesMaskedPointsRandom.end(), randomEngine);
    // // build quad tree
    quadtree::Box<float> domain = quadtree::Box<float>(0.0f, 0.0f, image.cols, image.rows);
    auto GetZeroSizeBox = [](Valid2DPoint* vpoint){
        return vpoint->zeroSizeBox;
    };
    auto thisQuadtree = quadtree::Quadtree<Valid2DPoint*, decltype(GetZeroSizeBox)>(domain, GetZeroSizeBox);
    size_t countNode = 0;
    std::vector<Valid2DPoint> allPoints;
    allPoints.reserve(coordinateMaskedPoints.size());
    for (size_t indexMaskedPoints : indicesMaskedPointsRandom){
        allPoints.push_back(GenerateValid2DPointFromCoordinates(coordinateMaskedPoints[indexMaskedPoints].x, coordinateMaskedPoints[indexMaskedPoints].y, countNode));
        thisQuadtree.add(&allPoints[allPoints.size() - 1]);
        countNode ++;
    }
    // // get top N points
    std::vector<Valid2DPoint*> pointQueue;
    thisQuadtree.ReturnDistributedNPoints(numFeaturePoints, pointQueue);
    if (featurePointsX.size() == 0){
        featurePointsX = Eigen::MatrixXi::Zero(1, numFeaturePoints);
    }
    if (featurePointsY.size() == 0){
        featurePointsY = Eigen::MatrixXi::Zero(1, numFeaturePoints);
    }
    for (size_t indexPoint=0; indexPoint < pointQueue.size(); indexPoint++){
        featurePointsX(0, indexPoint) = (pointQueue[indexPoint])->zeroSizeBox.left;
        featurePointsY(0, indexPoint) = (pointQueue[indexPoint])->zeroSizeBox.top;
    }
    featureVector = _ComputeImagePatchFeatureVector(imageLaplacian, gradMagnitudeThreshold, featurePointsX, featurePointsY);
    return;
}

Eigen::MatrixXi _ComputeImagePatchFeatureVector(
    const cv::Mat& inputImage,
    const float gradMagnitudeThreshold,
    const Eigen::MatrixXi& keyPointsX,
    const Eigen::MatrixXi& keyPointsY
){
    Eigen::Matrix<int, 1, Eigen::Dynamic> inputImageFeatureVector(1, keyPointsX.cols());
    size_t imageH = inputImage.rows;
    size_t imageW = inputImage.cols;
    for (size_t indexKeyPoint = 0; indexKeyPoint < keyPointsX.cols(); indexKeyPoint++){
        int keyPointX = keyPointsX(0, indexKeyPoint);
        int keyPointY = keyPointsY(0, indexKeyPoint);
        cv::Rect localPatchRoi(std::max(keyPointX - 2, 0), std::max(keyPointY - 2, 0), 5, 5);  // Note: if the patch too close to image border, it may be moved a little bit to make sure the patch can be complete.
        cv::Mat localPatch = inputImage(localPatchRoi);
        const int binNumberOfThisPatch = _ComputeQuantizedGradientOrientation(localPatch, gradMagnitudeThreshold);
        inputImageFeatureVector(0, indexKeyPoint) = binNumberOfThisPatch;
    }
    return inputImageFeatureVector;
}

