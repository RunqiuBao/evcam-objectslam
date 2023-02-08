import cv2
import os
import numpy
import json

import logging
logger = logging.getLogger(__name__)


class EvlinemodTemplate(object):
    _image = None
    _featureVector = None
    _featurePointsX = None
    _featurePointsY = None
    _imageW = None
    _imageH = None

    def __init__(self, image, featureVector, featurePointsX, featurePointsY):
        self._image = image
        self._imageH = image.shape[0]
        self._imageW = image.shape[1]
        self._featureVector = featureVector
        self._featurePointsX = featurePointsX
        self._featurePointsY = featurePointsY

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
            inputFeatureVector[indexPoint] = EvlinemodTemplate.ComputeQuantizedGradientOrientation(localPatch, gradMagnitudeThreshold=gradMagnitudeThreshold)
        return inputFeatureVector


class EventLinemodTemplateManager(object):
    _templateInfo = None
    
    def __init__(self, templatePath) -> None:
        '''
        This class handles templates, 
        - extract linemodality, extract mask, sample feature points on the template.
        - handle the scales, viewpoints of the templates.
        Args: 
            templatePath: the path to a folder of templates, including *.png and templateInfos.json (camInObjectTransformation).
        '''
        with open(os.path.join(templatePath, 'templateInfos.json'), 'r') as jsonFile:
            self._templateInfo = json.load(jsonFile)

        from IPython import embed; print('here!'); embed()
        

if __name__ == "__main__":
    aa = EventLinemodTemplateManager('/media/runqiu/data/eventLinemodDatasets/templates/')