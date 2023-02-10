import cv2
import os
import numpy
import json
import queue

from quadtree import Point, Rect, QuadTree

import logging
logger = logging.getLogger(__name__)


class EventLinemodTemplate(object):
    _image = None
    _simulationCamInObjectTransform = None
    _featureVector = None
    _featurePointsX = None
    _featurePointsY = None
    _imageW = None
    _imageH = None

    def __init__(self, image, simulationCamInObjectTransform, resize=1.0):
        simulationCamInObjectTransform[:3, 3] /= resize
        self._simulationCamInObjectTransform = simulationCamInObjectTransform
        self._image = cv2.resize(image.astype('uint8'), dsize=None, fx=resize, fy=resize, interpolation=cv2.INTER_CUBIC)
        self._imageH = image.shape[0]
        self._imageW = image.shape[1]
        templateMask = self._image >= 10
        self._image[~templateMask] = 0
        self._featurePointsX, self._featurePointsY, self._featureVector = EventLinemodTemplate.GetFeatureVector(self._image)

    @staticmethod
    def GetFeatureVector(image, numFeaturePoints=64, maxNumPointsQuadtreeNode=10, gradMagnitudeThreshold=100):
        '''
        Randomized and distributed N feature points forming as a featured vector
        '''
        # sample feature points
        imageLaplacian = cv2.Laplacian(image, cv2.CV_32F, ksize=3)
        imageLaplacianMask = numpy.where(numpy.abs(imageLaplacian) >= 40, 255, 0)
        kernels = []
        kernels.append(numpy.ones((3, 3), numpy.uint8))
        imageLaplacianMask = cv2.morphologyEx(imageLaplacianMask.astype('uint8'), cv2.MORPH_CLOSE, kernels[0])
        # # select n feature points
        coordinateMaskedPoints = numpy.where(imageLaplacianMask > 0)
        indexMaskedPointsRandom = numpy.random.default_rng().choice(coordinateMaskedPoints[0].shape[0], size=coordinateMaskedPoints[0].shape[0], replace=False)
        # build quad tree
        h, w = image.shape
        domain = Rect(w // 2, h // 2, w, h)
        qtree = QuadTree(domain, maxNumPointsQuadtreeNode)
        for index in indexMaskedPointsRandom:
            featurePoint = Point(coordinateMaskedPoints[1][index], coordinateMaskedPoints[0][index], 1.0)
            qtree.insert(featurePoint)
        # get top N points
        nodeQueue = EventLinemodTemplate.ReturnDistributedTopScoreNPoints(qtree, N=numFeaturePoints)
        distributedPointsX, distributedPointsY = [], []
        for i in range(nodeQueue.qsize()):
            thisNode = nodeQueue.get()
            foundPointsX, foundPointsY, foundPointsScore = [], [], []
            foundPointsX, foundPointsY = thisNode.getAllPoints(foundPointsX, foundPointsY, foundPointsScore)[:2]
            distributedPointsX.append(foundPointsX[0])
            distributedPointsY.append(foundPointsY[0])
        distributedPointsX, distributedPointsY = numpy.array(distributedPointsX), numpy.array(distributedPointsY)
        # compute feature vector
        featureVector = numpy.zeros_like(distributedPointsY)
        for indexPoint, localCenter in enumerate(zip(distributedPointsX, distributedPointsY)):
            localPatch = imageLaplacianMask[localCenter[0] - 2:localCenter[0] + 3, localCenter[1] - 2:localCenter[1] + 3]
            featureVector[indexPoint] = EventLinemodTemplate.ComputeQuantizedGradientOrientation(localPatch, gradMagnitudeThreshold=gradMagnitudeThreshold)
        return distributedPointsX, distributedPointsX, featureVector

    ## BFS
    @staticmethod
    def ReturnDistributedTopScoreNPoints(rootNode, N=600):
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
        response = numpy.argmin(numpy.abs(numpy.repeat(numpy.arctan(grad), 8, axis=2) - sectorCenterTangentValues)) + 1
        response[numpy.where(gradMagnitude < gradMagnitudeThreshold)] = 0
        ## (2) nested for loop
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

    def ComputeImagePatchFeatureVector(self, imagePatch, gradMagnitudeThreshold=100):
        inputFeatureVector = numpy.zeros_like(self._featureVector)
        for indexPoint, localCenter in enumerate(zip(self._featurePointsY, self._featurePointsX)):
            localPatch = imagePatch[localCenter[0] - 2:localCenter[0] + 3, localCenter[1] - 2:localCenter[1] + 3]
            inputFeatureVector[indexPoint] = EventLinemodTemplate.ComputeQuantizedGradientOrientation(localPatch, gradMagnitudeThreshold=gradMagnitudeThreshold)
        return inputFeatureVector


class EventLinemodTemplateManager(object):
    _templateInfo = None
    
    def __init__(self, templateInfo, templateData) -> None:
        '''
        This class handles templates, 
        - extract linemodality, extract mask, sample feature points on the template.
        - handle the scales, viewpoints of the templates.
        Args:
            templateData: list of images
            templateInfo: list of templateId and simulationCamInObjectTransformation
        '''


        from IPython import embed; print('here!'); embed()
        

if __name__ == "__main__":
    aa = EventLinemodTemplateManager('/media/runqiu/data/eventLinemodDatasets/templates/')