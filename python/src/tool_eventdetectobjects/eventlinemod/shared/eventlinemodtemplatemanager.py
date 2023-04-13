import cv2
import os
import numpy
import json
import queue
import copy

from .quadtree import Point, Rect, QuadTree

import logging
logger = logging.getLogger(__name__)


class EventLinemodTemplate(object):
    _image = None
    _sparsity = None
    _simulationCamInObjectTransform = None
    _templateId = None
    _featureVector = None
    _featurePointsX = None
    _featurePointsY = None
    _imageW = None
    _imageH = None
    _scaleFactorCache = None

    def __init__(self, image, simulationCamInObjectTransform, templateId, resize=0.33):
        simulationCamInObjectTransform[:3, 3] /= resize
        self._simulationCamInObjectTransform = simulationCamInObjectTransform
        self._templateId = templateId
        self._image = cv2.resize(image.astype('uint8'), dsize=None, fx=resize, fy=resize, interpolation=cv2.INTER_CUBIC)
        self._imageH = self._image.shape[0]
        self._imageW = self._image.shape[1]
        templateMask = self._image >= 10
        self._image[~templateMask] = 0
        self._featurePointsX, self._featurePointsY, self._featureVector, self._sparsity = EventLinemodTemplate.GetFeatureVector(self._image)
        self._scaleFactorCache = resize

    def RescaleThisTemplate(self, scaleFactor):  # scaleFactor is a absolute scale.
        scaleFactorRelative = scaleFactor / self._scaleFactorCache  # scaleFactorRelative is the relative scale , which we need to perform image transformation.
        self._simulationCamInObjectTransform[:3, 3] /= scaleFactorRelative
        self._image = cv2.resize(self._image.astype('uint8'), dsize=None, fx=scaleFactorRelative, fy=scaleFactorRelative, interpolation=cv2.INTER_CUBIC)
        self._imageH = self._image.shape[0]
        self._imageW = self._image.shape[1]
        self._featurePointsX, self._featurePointsY, self._featureVector, self._sparsity = EventLinemodTemplate.GetFeatureVector(self._image)
        self._scaleFactorCache = scaleFactor

    @property
    def originalH(self):
        return self._image.shape[0]

    @property
    def originalW(self):
        return self._image.shape[1]

    @property
    def simulationCamInObjectTransform(self):
        return self._simulationCamInObjectTransform

    @property
    def sparsity(self):
        return self._sparsity

    @property
    def scaleFactor(self):
        return self._scaleFactorCache

    @staticmethod
    def GetFeatureVector(image, numFeaturePoints=64, maxNumPointsQuadtreeNode=10, gradMagnitudeThreshold=100, isNoiseThreshold=40):
        '''
        Randomized and distributed N feature points forming as a featured vector
        '''
        # sample feature points
        imageLaplacian = cv2.Laplacian(image, cv2.CV_32F, ksize=3)
        # ## save plt fig in image (without axis and margin, only contents)
        # import matplotlib.pyplot as plt
        # fig = plt.figure(frameon=False)
        # fig.set_size_inches(184, 331)
        # ax = plt.Axes(fig, [0., 0., 1., 1.])
        # ax.set_axis_off()
        # fig.add_axes(ax)
        # ax.imshow(imageLaplacian, cmap='gray', aspect='auto')
        # plt.savefig('/media/runqiu/data/eventLinemodDatasets/templatesInLaplacian/{}.png'.format(str(countTemplate).zfill(6)), dpi=1)

        imageLaplacianMask = numpy.where(numpy.abs(imageLaplacian) >= isNoiseThreshold, 255, 0)
        kernels = []
        kernels.append(numpy.ones((3, 3), numpy.uint8))
        imageLaplacianMask = cv2.morphologyEx(imageLaplacianMask.astype('uint8'), cv2.MORPH_CLOSE, kernels[0])
        templateSparsity = (imageLaplacianMask).sum() / (imageLaplacianMask.shape[0] * imageLaplacianMask.shape[1] * 255)
        # # select n feature points
        coordinateMaskedPoints = numpy.where(imageLaplacianMask > 0)        
        indexMaskedPointsRandom = numpy.random.default_rng().choice(coordinateMaskedPoints[0].shape[0], size=coordinateMaskedPoints[0].shape[0], replace=False)
        # adjust numFeaturePoints
        if indexMaskedPointsRandom.size < numFeaturePoints * maxNumPointsQuadtreeNode:
            marginFactor = 2
            maxNumPointsQuadtreeNode = indexMaskedPointsRandom.size // numFeaturePoints // marginFactor
            if maxNumPointsQuadtreeNode < 1:
                raise ValueError("template too small and could not extract enough feature points.")
        # build quad tree
        h, w = image.shape
        domain = Rect(w // 2, h // 2, w, h)
        qtree = QuadTree(domain, maxNumPointsQuadtreeNode)
        for index in indexMaskedPointsRandom:
            featurePoint = Point(coordinateMaskedPoints[1][index], coordinateMaskedPoints[0][index], 1.0)
            qtree.insert(featurePoint)
        # get top N points
        nodeQueue = EventLinemodTemplate.ReturnDistributedNPoints(qtree, N=numFeaturePoints)
        distributedPointsX, distributedPointsY = [], []
        for i in range(nodeQueue.qsize()):
            thisNode = nodeQueue.get()
            foundPointsX, foundPointsY, foundPointsScore = [], [], []
            foundPointsX, foundPointsY = thisNode.getAllPoints(foundPointsX, foundPointsY, foundPointsScore)[:2]
            distributedPointsX.append(foundPointsX[0])
            distributedPointsY.append(foundPointsY[0])
        distributedPointsX, distributedPointsY = numpy.array(distributedPointsX), numpy.array(distributedPointsY)
        # compute feature vector
        featureVector = EventLinemodTemplate._ComputeImagePatchFeatureVector(imageLaplacianMask, gradMagnitudeThreshold, distributedPointsX, distributedPointsY)
        return distributedPointsX, distributedPointsY, featureVector, templateSparsity

    ## BFS
    @staticmethod
    def ReturnDistributedNPoints(rootNode, N=600):
        if N > len(rootNode):
            N = len(rootNode)
        nodeQueue = queue.Queue()
        nodeQueue.put(rootNode)
        while nodeQueue.qsize() < N:
    #         if nodeQueue.qsize() == 1214:
    #             import pdb; pdb.set_trace()
            firstNode = nodeQueue.get()
            if firstNode.divided:
                if len(firstNode.nw) != 0:
                    nodeQueue.put(firstNode.nw)
                if len(firstNode.ne) != 0:
                    nodeQueue.put(firstNode.ne)
                    if nodeQueue.qsize() >= N:
                        break
                if len(firstNode.se) != 0:
                    nodeQueue.put(firstNode.se)
                    if nodeQueue.qsize() >= N:
                        break
                if len(firstNode.sw) != 0:
                    nodeQueue.put(firstNode.sw)
            else:
                if len(firstNode) != 0:
                    nodeQueue.put(firstNode)
        return nodeQueue

    @property
    def templateId(self):
        return self._templateId

    @property
    def imageH(self):
        return self._imageH

    @property
    def imageW(self):
        return self._imageW

    @property
    def featureVector(self):
        return self._featureVector

    @staticmethod
    def ComputeQuantizedGradientOrientation(imagePatch, numSector=8, gradMagnitudeThreshold=100):
        gray_x = cv2.Sobel(imagePatch, cv2.CV_32F, 1, 0, ksize=3)[1:-1, 1:-1]
        gray_y = cv2.Sobel(imagePatch, cv2.CV_32F, 0, 1, ksize=3)[1:-1, 1:-1]
        grad = gray_y / gray_x
        # (1) numpy
        gradMagnitude = numpy.linalg.norm(numpy.concatenate([gray_x[:, :, numpy.newaxis], gray_y[:, :, numpy.newaxis]], axis=2), axis=2)
        sectorCenterTangentValues = numpy.tile(numpy.array([0.5 * numpy.pi / numSector, 1.5 * numpy.pi / numSector, 2.5 * numpy.pi / numSector, 3.5 * numpy.pi / numSector, -3.5 * numpy.pi / numSector, -2.5 * numpy.pi / numSector, -1.5 * numpy.pi / numSector, -0.5 * numpy.pi / numSector]), (grad.shape[0], grad.shape[1], 1))
        response = numpy.argmin(numpy.abs(numpy.repeat(numpy.arctan(grad)[:, :, numpy.newaxis], 8, axis=2) - sectorCenterTangentValues), axis=2) + 1
        response[numpy.where(gradMagnitude < gradMagnitudeThreshold)] = 0
        response[numpy.isnan(grad)] = 0
        # # (2) nested for loop
        # response = numpy.zeros_like(grad, dtype='int')
        # for px in range(3):
        #     for py in range(3):
        #         if numpy.linalg.norm([gray_x[py, px], gray_y[py, px]]) < gradMagnitudeThreshold:
        #             response[py, px] = 0
        #         elif grad[py, px] > 0 and grad[py, px] <= numpy.tan(numpy.pi / numSector):
        #             response[py, px] = 1
        #         elif grad[py, px] > numpy.tan(numpy.pi / numSector) and grad[py, px] <= numpy.tan(numpy.pi * 2 / numSector):
        #             response[py, px] = 2
        #         elif grad[py, px] > numpy.tan(numpy.pi * 2 / numSector) and grad[py, px] <= numpy.tan(numpy.pi * 3 / numSector):
        #             response[py, px] = 3
        #         elif grad[py, px] > numpy.tan(numpy.pi * 3 / numSector):
        #             response[py, px] = 4
        #         elif grad[py, px] <= numpy.tan(-numpy.pi * 3 / numSector):
        #             response[py, px] = 5
        #         elif grad[py, px] > numpy.tan(-numpy.pi * 3 / numSector) and grad[py, px] <= numpy.tan(-numpy.pi * 2 / numSector):
        #             response[py, px] = 6
        #         elif grad[py, px] > numpy.tan(-numpy.pi * 2 / numSector) and grad[py, px] <= numpy.tan(-numpy.pi / numSector):
        #             response[py, px] = 7
        #         elif grad[py, px] > numpy.tan(-numpy.pi / numSector) and grad[py, px] <= 0:
        #             response[py, px] = 8
        binCount = numpy.bincount(response.reshape(9,))

        mainScore = numpy.argmax(binCount)
        # print('mainScore: \n', mainScore)
        # if mainScore == 0 and response.sum() > 0:
        #     binCount[mainScore] = 0
        #     print('binCount: \n', binCount)
        #     mainScore = numpy.argmax(binCount)
        return mainScore

    def ComputeImagePatchFeatureVector(self, imagePatch, gradMagnitudeThreshold=100, debugInfo=None):
        return EventLinemodTemplate._ComputeImagePatchFeatureVector(imagePatch, gradMagnitudeThreshold, self._featurePointsX, self._featurePointsY, debugInfo)

    @staticmethod
    def _ComputeImagePatchFeatureVector(imagePatch, gradMagnitudeThreshold, featurePointsX, featurePointsY, debugInfo=None):
        inputFeatureVector = numpy.zeros_like(featurePointsY)
        hh, ww = imagePatch.shape
        if debugInfo is not None:
            debugInfo['xyFeatures'] = numpy.zeros((featurePointsY.size, 2))
            debugInfo['displayImage'] = copy.deepcopy(imagePatch)
            color = (0,)
            thickness = 1
            vizGrads = {
                0: numpy.array([0, 0]),
                1: numpy.array([1, 5]),
                2: numpy.array([2, 4]),
                3: numpy.array([4, 2]),
                4: numpy.array([5, 1]),
                5: numpy.array([5, -1]),
                6: numpy.array([4, -2]),
                7: numpy.array([2, -4]),
                8: numpy.array([1, -5])
            }
        for indexPoint, localCenter in enumerate(zip(featurePointsY, featurePointsX)):
            localPatch = imagePatch[max(localCenter[0] - 2, 0):min(localCenter[0] + 3, hh), max(localCenter[1] - 2, 0):min(localCenter[1] + 3, ww)]
            if localPatch.shape != (5, 5):
                try:
                    localPatch = numpy.pad(localPatch, ((0, 5 - localPatch.shape[0]), (0, 5 - localPatch.shape[1])), 'edge')
                except:
                    from IPython import embed; logger.debug('here!'); embed()

            inputFeatureVector[indexPoint] = EventLinemodTemplate.ComputeQuantizedGradientOrientation(localPatch, gradMagnitudeThreshold=gradMagnitudeThreshold)
            if debugInfo is not None:
                debugInfo['xyFeatures'][indexPoint, 0] = localCenter[1]
                debugInfo['xyFeatures'][indexPoint, 1] = localCenter[0]
                # cv2.circle(debugInfo['displayImage'], (localCenter[1], localCenter[0]), 5, (0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
                debugInfo['displayImage'] = cv2.arrowedLine(debugInfo['displayImage'], numpy.array(localCenter)[::-1], numpy.array(localCenter)[::-1] + vizGrads[inputFeatureVector[indexPoint]], color, thickness) 
        return inputFeatureVector


class EventLinemodTemplateManager(object):
    _templateList = None
    _templateIdList = None
    _scale = None
    # make this class an iterator
    _templatePtr = None
    
    def __init__(self, templateInfos, templateData) -> None:
        '''
        This class handles templates, 
        - extract linemodality, extract mask, sample feature points on the template.
        - handle the scales, viewpoints of the templates.
        Args:
            templateData: list of images
            templateInfo: list of templateId and simulationCamInObjectTransformation
        '''
        self._templateList = []
        self._templateIdList = []
        for templateInfo, templateDatum in zip(templateInfos, templateData):
            self._templateList.append(EventLinemodTemplate(templateDatum, templateInfo['camInObjectTransformation'], templateInfo['templId']))
            self._templateIdList.append(templateInfo['templId'])
        logger.debug("templateManager initialized with %d templates.", len(self._templateIdList))

    def GetObjectDistance(self, templateId, scale, extraFactor):
        thisTemplate = self._templateManager._templateList[templateId]
        templateScaleCache = thisTemplate.scaleFactor
        thisTemplate.RescaleThisTemplate(scale)
        objectDistance = numpy.linalg.norm(thisTemplate.simulationCamInObjectTransform[:3, 3]) / extraFactor
        thisTemplate.RescaleThisTemplate(templateScaleCache)
        return objectDistance

    def GetTemplate(self, templateId):
        return self._templateList[self._templateIdList.index(templateId)]

    @property
    def scale(self):
        return self._scale

    @property
    def length(self):
        return len(self._templateList)

    @scale.setter
    def scale(self, value):
        if value < 0:
            raise ValueError("scale cannot be negative.")
        self._scale = value

    # make this class an iterator
    def __iter__(self):
        self._templatePtr = 0
        return self

    def __next__(self):
        if self._templatePtr == len(self._templateList):
            self._templatePtr = None
            raise StopIteration()
        template = self._templateList[self._templatePtr]
        if template.scaleFactor == self._scale:
            pass
        else:            
            template.RescaleThisTemplate(self._scale)
        self._templatePtr += 1
        return template


if __name__ == "__main__":
    aa = EventLinemodTemplateManager('/media/runqiu/data/eventLinemodDatasets/templates/')
