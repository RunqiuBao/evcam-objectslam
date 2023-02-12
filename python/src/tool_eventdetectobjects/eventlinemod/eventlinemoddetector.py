import numpy
import time
import copy
import cv2

from .shared.nms import DoNonMaxSuppression

import logging
logger = logging.getLogger(__name__)


class EventLinemodDetection():
    _x = None
    _y = None
    _templateId = None
    _score = None
    _scale = None
    _bbox = None

    def __init__(self, x, y, templateId, score, scale, bbox) -> None:
        self._x, self._y, self._templateId, self._score, self._scale, self._bbox = x, y, templateId, score, scale, bbox


class EventLinemodDetector(object):
    _templateManager = None
    _templateResponseThreshold = None

    def __init__(self, templateManager, templateResponseThreshold=100) -> None:
        self._templateManager = templateManager
        self._templateResponseThreshold = templateResponseThreshold

    def DetectTemplatesSemiScaleInvariant(self, inputFrame, minScale=0.6944, maxScale=1.44, scaleMultiplier=1.2, scanStep=4, isShow=False):
        starttime = time.time()
        logger.debug("======== detection function start ========")
        # detect search
        uncenteredSceneImage = (inputFrame - inputFrame.min()).astype('float32')

        detectionList = []
        detectionBBoxes = []
        detectionScores = []
        isShow = True
        if isShow:
            imageDisplay = cv2.cvtColor(copy.deepcopy(uncenteredSceneImage).astype('uint8'), cv2.COLOR_GRAY2RGB).astype('float')
            imageDisplay = (imageDisplay * 255 / imageDisplay.max()).astype('uint8')
        currentScale = minScale
        while currentScale <= maxScale:
            self._templateManager.scale = currentScale
            for oneTemplate in self._templateManager:
                responseMat = numpy.zeros(
                    (int(numpy.ceil((uncenteredSceneImage.shape[0] - oneTemplate.imageH) / scanStep)), int(numpy.ceil((uncenteredSceneImage.shape[1] - oneTemplate.imageW) / scanStep))), 
                    dtype='int'
                )

                for xx in range(0, uncenteredSceneImage.shape[1] - oneTemplate.imageW, scanStep):
                    for yy in range(0, uncenteredSceneImage.shape[0] - oneTemplate.imageH, scanStep):
                        scanWindow = uncenteredSceneImage[yy:yy + oneTemplate.imageH, xx:xx + oneTemplate.imageW]
                        windowFeatureVector = oneTemplate.ComputeImagePatchFeatureVector(scanWindow, gradMagnitudeThreshold=1)
                        pixelResponse = numpy.zeros_like(windowFeatureVector)
                        zeroMask = numpy.zeros_like(windowFeatureVector)
                        zeroMask[numpy.where(windowFeatureVector == 0)] = 1
                        zeroMask = zeroMask.astype('bool')
                        pixelResponse[zeroMask] = 8
                        pixelResponse[~zeroMask] = numpy.abs(windowFeatureVector[~zeroMask] - oneTemplate.featureVector[~zeroMask])
                        responseMat[yy // scanStep, xx // scanStep] = pixelResponse.sum()
                        if responseMat[yy // scanStep, xx // scanStep] <= self._templateResponseThreshold:
                            detectionList.append(EventLinemodDetection(yy, xx, oneTemplate.templateId, responseMat.min(), currentScale, [yy, xx, yy + oneTemplate.imageW, xx + oneTemplate.imageH]))
                            detectionBBoxes.append([yy, xx, yy + oneTemplate.imageW, xx + oneTemplate.imageH])
                            detectionScores.append(responseMat[yy // scanStep, xx // scanStep])
                logger.debug("finished scan templateId %s at scale %f, min response: %d", oneTemplate.templateId, currentScale, responseMat.min())
                # indexMin = numpy.unravel_index(numpy.argmin(responseMat), responseMat.shape)
                # detectionList.append(EventLinemodDetection(indexMin[1], indexMin[0], oneTemplate.templateId, responseMat.min(), currentScale, [indexMin[1], indexMin[0], indexMin[1] + oneTemplate.imageW, indexMin[0] + oneTemplate.imageH]))
            currentScale *= scaleMultiplier
        logger.debug("%d raw detections...", len(detectionList))
        detectionList, detectionBBoxes, detectionScores = DoNonMaxSuppression(detectionList, detectionBBoxes, detectionScores)
        if isShow:
            for indexDetection, detectionBBox in enumerate(detectionBBoxes):
                cv2.rectangle(imageDisplay, (detectionBBox[0], detectionBBox[1]), (detectionBBox[0] + oneTemplate.imageW, detectionBBox[1] + oneTemplate.imageH), (255, 0, 0), 2)
                cv2.putText(imageDisplay, str(detectionList[indexDetection]._scale), (detectionBBox[0] + 10, detectionBBox[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        logger.debug("======== detection finished in {} secs, totally {} overlap-free detections ========".format(time.time() - starttime, len(detectionList)))
