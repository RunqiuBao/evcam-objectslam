import numpy
import time
import copy
import cv2
import os
import pickle
from datetime import datetime
import json

from .shared.nms import DoNonMaxSuppression
from ..commonutils.ioutils import LmdbWriter
from ..commonutils.datatypeutils import MyJsonEncoder, MyJsonDecoder

import logging
logger = logging.getLogger(__name__)


def RescaleImage(inputImage, targetH, targetW):
    return cv2.resize(inputImage, dsize=(targetW, targetH))


class EventLinemodDetection():
    _x = None
    _y = None
    _templateId = None
    _score = None
    _scale = None
    _bbox = None

    def __init__(self, x, y, templateId, score, scale, bbox) -> None:
        self._x, self._y, self._templateId, self._score, self._scale, self._bbox = x, y, templateId, score, scale, bbox

    @property
    def templateId(self):
        return self._templateId

    @property
    def score(self):
        return self._score

    @property
    def scale(self):
        return self._scale

    @property
    def bbox(self):
        return self._bbox


class EventLinemodDetector(object):
    _templateManager = None
    _templateResponseThreshold = None

    def __init__(self, templateManager, templateResponseThreshold=100) -> None:
        self._templateManager = templateManager
        self._templateResponseThreshold = templateResponseThreshold

    def DetectTemplatesSemiScaleInvariant(self, inputFrame, minScale=0.6944, maxScale=1.44, scaleMultiplier=1.2, scanStep=4, isTooSparseThreshold=0.5, isShow=False, debugPathRoot=None, dataSaveRoot=None, frameIndex=0):
        starttime = time.time()
        logger.debug("======== detection function start ========")
        # detect search
        activePixelMask = numpy.zeros_like(inputFrame)
        activePixelMask[numpy.where(numpy.abs(inputFrame)> 0)] = 1
        uncenteredSceneImage = (inputFrame - inputFrame.min()).astype('float32')

        detectionList = []
        detectionBBoxes = []
        detectionScores = []
        isShow = True
        currentScale = minScale

        # save training data
        dataWriters = []
        for indexTemplate in range(self._templateManager.length):
            os.makedirs(os.path.join(dataSaveRoot, 'template_' + str(indexTemplate)), exist_ok=True)
            dataWriters.append(LmdbWriter(os.path.join(dataSaveRoot, 'template_' + str(indexTemplate))))
        while currentScale <= maxScale:
            self._templateManager.scale = currentScale
            for oneTemplate in self._templateManager:
                responseMat = numpy.zeros(
                    (int(numpy.ceil((uncenteredSceneImage.shape[0] - oneTemplate.imageH) / scanStep)), int(numpy.ceil((uncenteredSceneImage.shape[1] - oneTemplate.imageW) / scanStep))), 
                    dtype='int'
                )
                for xx in range(0, uncenteredSceneImage.shape[1] - oneTemplate.imageW, scanStep):
                    for yy in range(0, uncenteredSceneImage.shape[0] - oneTemplate.imageH, scanStep):
                        if activePixelMask[yy:yy + oneTemplate.imageH, xx:xx + oneTemplate.imageW].sum() < (isTooSparseThreshold * oneTemplate.sparsity):
                            continue  # this scan window is sparse
                        scanWindow = uncenteredSceneImage[yy:yy + oneTemplate.imageH, xx:xx + oneTemplate.imageW]
                        debugInfo = {}
                        windowFeatureVector = oneTemplate.ComputeImagePatchFeatureVector(scanWindow, gradMagnitudeThreshold=1, debugInfo=debugInfo)
                        pixelResponse = numpy.zeros_like(windowFeatureVector)
                        zeroMask = numpy.zeros_like(windowFeatureVector)
                        zeroMask[numpy.where(windowFeatureVector == 0)] = 1
                        zeroMask = zeroMask.astype('bool')
                        pixelResponse[zeroMask] = 8
                        pixelResponse[~zeroMask] = numpy.abs(windowFeatureVector[~zeroMask] - oneTemplate.featureVector[~zeroMask])
                        responseMat[yy // scanStep, xx // scanStep] = pixelResponse.sum()
                        if responseMat[yy // scanStep, xx // scanStep] <= self._templateResponseThreshold:
                            detectionList.append(EventLinemodDetection(xx, yy, oneTemplate.templateId, responseMat[yy // scanStep, xx // scanStep], currentScale, [xx, yy, xx + oneTemplate.imageW, yy + oneTemplate.imageH]))
                            detectionBBoxes.append([xx, yy, xx + oneTemplate.imageW, yy + oneTemplate.imageH])
                            detectionScores.append(responseMat[yy // scanStep, xx // scanStep])
                    logger.debug("finished one column {}".format(xx))    
                logger.debug("finished scan templateId %s at scale %f in %f secs, min response: %d", oneTemplate.templateId, currentScale, time.time() - starttime, responseMat.min())
                # indexMin = numpy.unravel_index(numpy.argmin(responseMat), responseMat.shape)
                # detectionList.append(EventLinemodDetection(indexMin[1], indexMin[0], oneTemplate.templateId, responseMat.min(), currentScale, [indexMin[1], indexMin[0], indexMin[1] + oneTemplate.imageW, indexMin[0] + oneTemplate.imageH]))
            currentScale *= scaleMultiplier
        logger.debug("%d raw detections...", len(detectionList))

        detectionListOverlapFree, detectionBBoxesOverlapFree, detectionScoresOverlapFree = detectionList, detectionBBoxes, detectionScores #DoNonMaxSuppression(detectionList, detectionBBoxes, detectionScores, overlapThreshold=0.3)

        if isShow:
            imageDisplay = cv2.cvtColor(copy.deepcopy(uncenteredSceneImage).astype('uint8'), cv2.COLOR_GRAY2RGB).astype('float')
            imageDisplay = (imageDisplay * 255 / imageDisplay.max()).astype('uint8')
            try:
                imageDisplayForSave = cv2.cvtColor(copy.deepcopy(uncenteredSceneImage).astype('uint8'), cv2.COLOR_GRAY2RGB).astype('float')
                imageDisplayForSave = (imageDisplayForSave * 255 / imageDisplayForSave.max()).astype('uint8')
                countTemplates = [0 for indexTemplate in range(len(self._templateManager._templateList))]
                for indexDetection, detectionBBox in enumerate(detectionBBoxesOverlapFree):
                    # save training data
                    thisTemplate = self._templateManager.GetTemplate(detectionListOverlapFree[indexDetection].templateId)
                    # code = '%06d_%06d' % (frameIndex, indexDetection)
                    # code = code.encode()
                    # dataWriters[detectionListOverlapFree[indexDetection].templateId].write(code, RescaleImage(uncenteredSceneImage[detectionBBox[1]:detectionBBox[3], detectionBBox[0]:detectionBBox[2]].astype('uint8'), thisTemplate.originalH, thisTemplate.originalW))
                    countTemplates[detectionListOverlapFree[indexDetection].templateId] += 1
                    # imageName: indexDetection_score_distanceFromScale
                    distanceFromScale = 15.0 / 5.0 / detectionListOverlapFree[indexDetection]._scale # 5 is a factor for color cone model
                    cv2.imwrite('/home/runqiu/tmptmp/detections/frame{}_'.format(frameIndex) + str(indexDetection).zfill(6) + '_' + str(detectionListOverlapFree[indexDetection]._score) + '_' + "{:.2f}".format(distanceFromScale) + '.png', RescaleImage(imageDisplayForSave[detectionBBox[1]:detectionBBox[3], detectionBBox[0]:detectionBBox[2]], thisTemplate.originalH, thisTemplate.originalW))
                    cv2.rectangle(imageDisplay, (detectionBBox[0], detectionBBox[1]), (detectionBBox[2], detectionBBox[3]), (255, 0, 0), 2)
                    cv2.putText(imageDisplay, str(detectionListOverlapFree[indexDetection]._score) + '_' + "{:.2f}".format(distanceFromScale), (detectionBBox[0] + 10, detectionBBox[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                for dataWriter in dataWriters:
                    dataWriter.commitchange()
                    dataWriter.endwriting()
                dataInfosPath = os.path.join(dataSaveRoot, 'dataInfos.json')
                if os.path.exists(dataInfosPath):
                    with open(dataInfosPath, 'r') as cacheFile:
                        dataInfos = json.load(cacheFile, cls=MyJsonDecoder)
                else:
                    dataInfos = []
                dataInfos.append({
                    frameIndex: {
                        'numDetections': len(detectionBBoxesOverlapFree),
                        'numDetectionsEachTemplates': countTemplates
                    }
                })
                with open(os.path.join(dataSaveRoot, 'dataInfos.json'), 'w') as outfile:
                    json.dump(dataInfos, outfile, cls=MyJsonEncoder)
            except:
                from IPython import embed; print('here!'); embed()

        logger.debug("======== detection finished in {} secs, totally {} overlap-free detections ========".format(time.time() - starttime, len(detectionListOverlapFree)))

        try:
            self.LogResultState(detectionListOverlapFree, inputFrame, imageDisplay, debugPathRoot)
        except:
            from IPython import embed; print('here!'); embed()

        from IPython import embed; print('here!'); embed()

    def ValidateObjectInStereoPair(self, leftDetections, stereoCalib):
        pass

    def ValidateObjectByEarthPlane(self, detections, cameraHeight, cameraPitchAngle, ghostThreshold):
        '''
        Args:
            detections:
            cameraHeight: a rough height of the camera from ground
            cameraPitchAngle: a rough pitch angle of the camera.
            ghostThreshold: allowed distance of the object from ground surface.
        '''
        pass

    def LogResultState(self, detections, inputFrame, imageDisplay, debugPathRoot):
        logDict = {
            'detections': [],
            'inputFrame': inputFrame,
            'imageDisplay': imageDisplay
        }
        for indexDetection, detection in enumerate(detections):
            logDict['detections'].append(EventLinemodDetector.ConvertDetectionToDict(detection, indexDetection))
        now = datetime.now()
        now = now.strftime("%Y_%m_%dT%H-%M-%S")
        os.makedirs(os.path.join(debugPathRoot, now), exist_ok=True)
        with open(os.path.join(debugPathRoot, now, 'detectionResults.pkl'), 'wb') as f:
            pickle.dump(logDict, f)
        cv2.imwrite(os.path.join(debugPathRoot, now, 'imageDisplay.png'), imageDisplay)

    @staticmethod
    def ConvertDetectionToDict(detection, detectionId):
        detectionDict = {
            'detectionId': detectionId,
            'templateId': detection.templateId,
            'score': detection.score,
            'scale': detection.scale,
            'bbox': detection.bbox
        }
        return detectionDict