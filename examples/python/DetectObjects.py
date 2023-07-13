# import tool_eventdetectobjects.logger.commonlogging as commonlogging
# author: bao@robot.u-tokyo.ac.jp

from cgitb import small
import sys
import os
from urllib import response
import numpy
import cv2
import copy
import time
import glob
file_path = os.path.realpath(__file__)
sys.path.append(os.path.join(file_path[:len(file_path) - len(file_path.split('/')[-1])], '../../', 'python/src/'))

import tool_eventdetectobjects.logger.commonlogging as commonlogging
from tool_eventdetectobjects.io.file_format.eventsinethzformat import EventsInETHZFormat
from tool_eventdetectobjects.io.templateimporter import ReadFromTemplateFolder
from tool_eventdetectobjects.eventlinemod.shared.eventlinemodtemplatemanager import EventLinemodTemplateManager, EventLinemodTemplate
from tool_eventdetectobjects.eventlinemod.eventlinemoddetector import EventLinemodDetector

import logging
logger = logging.getLogger(__name__)


def DoDenseLineMod(image, gradMagnitudeThreshold=100):
    assert(len(image.shape) == 2)  # Note: must be grayscale image
    linemodMap = EventLinemodTemplate._ComputeDenseLinemod(image, gradMagnitudeThreshold)
    return linemodMap


def _OptimizeImageShape(image, optimalSize):
    if image is None or image.shape[0] == optimalSize[0] and image.shape[1] == optimalSize[1]:
        return image
    else:
        right, bottom = optimalSize[1] - image.shape[1], optimalSize[0] - image.shape[0]
        return cv2.copyMakeBorder(image, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=0)


def CrossCorrelationTemplateMatching(image, template):
    """
    Non-normalized cross-correlation
    """
    assert len(image.shape) == len(template.shape), 'image, template, and masks must have the same dimensions/channels'
    assert len(template.shape) == 2 and len(image.shape) == 2, 'this method only supports 2d arrays (single channel images)'
    assert template.shape[0] <= image.shape[0] and template.shape[1] <= image.shape[1] and template.shape[0] > 0 and template.shape[1] > 0, 'template larger than image is not supported!'
    assert image.dtype == numpy.dtype('float32') and template.dtype == numpy.dtype('float32'), 'only float32 images are allowed for precision'

    optimalSize = tuple([cv2.getOptimalDFTSize(size) for size in image.shape])
    imagePadded = _OptimizeImageShape(image, optimalSize)
    templatePadded = _OptimizeImageShape(template, optimalSize)

    responseShape = (image.shape[0] - template.shape[0] + 1, image.shape[1] - template.shape[1] + 1)
    imageFFT = cv2.dft(imagePadded)
    templateFFT = cv2.dft(templatePadded, nonzeroRows=template.shape[0])
    responseFFT = cv2.mulSpectrums(imageFFT, templateFFT, 0, conjB=True)
    response = cv2.idft(responseFFT, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT, nonzeroRows=responseShape[0])
    return response[:responseShape[0], :responseShape[1]]


def TransformSbn2Image(mysbn):
    aa = numpy.abs(mysbn)
    maxNonZero = cv2.findNonZero(aa)
    cc = mysbn
    if maxNonZero is not None:
        bb = aa[maxNonZero[:, 0, 1], maxNonZero[:, 0, 0]]
        mildMaxValue = numpy.mean(bb) + numpy.std(bb) * 3
        cc = cc * 127 / mildMaxValue
    cc = numpy.clip(cc + 127, 0, 255).astype('uint8')
    return cc


def ResizeBBoxAccordingToTemplate(bbox, oneTemplate):
    # distribute the error in x and y direction equally
    bboxWidth = bbox[4] - bbox[2]
    bboxHeight = bbox[5] - bbox[3]
    templateWidth = oneTemplate.templateW
    templateHeight = oneTemplate.templateH
    scaleFactor = (bboxWidth + bboxHeight) / (templateWidth + templateHeight)
    newTemplateW = scaleFactor * templateWidth
    newTemplateH = scaleFactor * templateHeight
    newbbox = [
        bbox[0],
        bbox[1],
        bbox[0] - 0.5 * newTemplateW,
        bbox[1] - 0.5 * newTemplateH,
        bbox[0] + 0.5 * newTemplateW,
        bbox[1] + 0.5 * newTemplateH
    ]
    return newbbox, scaleFactor

def SelectTemplateByCCTM(sceneImagePreprocessed, bbox, myTemplateManager, sceneImageShape):
    largestScore = 0
    indexLargestScore = 0
    debugInfo = {}
    for indexTemplate in range(myTemplateManager.length):
        oneTemplate = copy.deepcopy(myTemplateManager.GetTemplate(indexTemplate))
        # use eventlinemod to select template
        newbbox, additionalScaleFactorForTemplate = ResizeBBoxAccordingToTemplate(bbox, oneTemplate)
        oneTemplate.RescaleThisTemplate(oneTemplate.scaleFactor * additionalScaleFactorForTemplate)
        templateH, templateW = oneTemplate.templateH, oneTemplate.templateW
        minY = round(newbbox[1] - templateH * 0.5)
        maxY = minY + templateH
        minX = round(newbbox[0] - templateW * 0.5)
        maxX = minX + templateW
        if maxY > sceneImageShape[0]:
            biaxY = maxY - sceneImageShape[0]
            minY -= biaxY
            maxY -= biaxY
        if minY < 0:
            biaxY = 0 - minY
            minY += biaxY
            maxY += biaxY
        if maxX > sceneImageShape[1]:
            biaxX = maxX - sceneImageShape[1]
            minX -= biaxX
            maxX -= biaxX
        if minX < 0:
            biaxX = 0 - minX
            minX += biaxX
            maxX += biaxX
        scanWindowLineMod = sceneImagePreprocessed[minY:maxY, minX:maxX]
        templateImage = copy.deepcopy(oneTemplate._image)
        # med_val = numpy.median(templateImage)
        # sigma = 0.33  # 0.33
        # min_val = int(max(0, (1.0 - sigma) * med_val))
        # max_val = int(max(255, (1.0 + sigma) * med_val))
        # templateImage = cv2.Canny(templateImage, threshold1 = min_val, threshold2 = max_val)
        # templateImage = cv2.GaussianBlur(templateImage, (5, 5), 0)
        templateImageLinemod = DoDenseLineMod(templateImage, gradMagnitudeThreshold=100)

        response = CrossCorrelationTemplateMatching(scanWindowLineMod.astype('float32'), templateImageLinemod.astype('float32'))
        if response.max() > largestScore:
            largestScore = response.max()
            indexLargestScore = indexTemplate
            debugInfo['bestTemplateDebug'] = (templateImageLinemod * 31).astype('uint8')  # for visualization
        # print("indexTemplate {}, linemod score: {}".format(indexTemplate, pixelResponse.sum()))
    return indexLargestScore, largestScore, debugInfo


def SelectTemplateByEventLinemod(sceneImage, bbox, myTemplateManager, sceneImageShape):
    smallestScore = sys.float_info.max
    indexSmallestScore = 0
    debugInfo = {}
    for indexTemplate in range(myTemplateManager.length):
        oneTemplate = copy.deepcopy(myTemplateManager.GetTemplate(indexTemplate))
        # use eventlinemod to select template
        newbbox, additionalScaleFactorForTemplate = ResizeBBoxAccordingToTemplate(bbox, oneTemplate)
        oneTemplate.RescaleThisTemplate(oneTemplate.scaleFactor * additionalScaleFactorForTemplate)
        templateH, templateW = oneTemplate.templateH, oneTemplate.templateW
        minY = round(newbbox[1] - templateH * 0.5)
        maxY = minY + templateH
        minX = round(newbbox[0] - templateW * 0.5)
        maxX = minX + templateW
        if maxY > sceneImageShape[0]:
            biaxY = maxY - sceneImageShape[0]
            minY -= biaxY
            maxY -= biaxY
        if minY < 0:
            biaxY = 0 - minY
            minY += biaxY
            maxY += biaxY
        if maxX > sceneImageShape[1]:
            biaxX = maxX - sceneImageShape[1]
            minX -= biaxX
            maxX -= biaxX
        if minX < 0:
            biaxX = 0 - minX
            minX += biaxX
            maxX +=biaxX
        scanWindow = sceneImage[minY:maxY, minX:maxX]
        windowFeatureVector = oneTemplate.ComputeImagePatchFeatureVector(scanWindow, gradMagnitudeThreshold=5, debugInfo=debugInfo)
        pixelResponse = numpy.zeros_like(windowFeatureVector)
        zeroMask = numpy.zeros_like(windowFeatureVector)
        zeroMask[numpy.where(windowFeatureVector == 0)] = 1
        zeroMask = zeroMask.astype('bool')
        pixelResponse[zeroMask] = 8
        pixelResponse[~zeroMask] = numpy.abs(windowFeatureVector[~zeroMask] - oneTemplate.featureVector[~zeroMask])
        if pixelResponse.sum() < smallestScore:
            smallestScore = pixelResponse.sum()
            indexSmallestScore = indexTemplate
            debugInfo['bestTemplateDebug'] = debugInfo['displayImage']
        # print("indexTemplate {}, linemod score: {}".format(indexTemplate, pixelResponse.sum()))
    return indexSmallestScore, smallestScore, debugInfo


def DoTemplateRecognitionForDetections(detections, image, myTemplateManager, debugDir=None, imageName=None, scoreThreshold=300):
    imageHeight, imageWidth = image.shape
    imageForDebug = copy.deepcopy(image)
    templatedDetections = []
    image = cv2.GaussianBlur(image, (3, 3), 0)
    isUseCCTM = False
    if isUseCCTM:
        imageLineMod = DoDenseLineMod(image, gradMagnitudeThreshold=20)
        imageForDebug2 = (copy.deepcopy(imageLineMod) * 31).astype('uint8')  # for visualization
    else:
        imageForDebug2 = copy.deepcopy(image)

    for indexDetection, detectionOrigin in enumerate(detections):
        detection = detectionOrigin.split(' ')
        isNewLineAtEnd = False
        if detection[-1][-1] == '\n':
            detection[-1] = detection[-1][:-1]
            isNewLineAtEnd = True
        bboxWidth = float(detection[3]) * imageWidth
        bboxHeight = float(detection[4]) * imageHeight
        bbox = [
            float(detection[1]) * imageWidth,
            float(detection[2]) * imageHeight,
            float(detection[1]) * imageWidth - 0.5 * bboxWidth,
            float(detection[2]) * imageHeight - 0.5 * bboxHeight,
            float(detection[1]) * imageWidth + 0.5 * bboxWidth,
            float(detection[2]) * imageHeight + 0.5 * bboxHeight
        ]
        indexSmallestShapeDifference, smallestScore, debugInfo = SelectTemplateByEventLinemod(imageLineMod if isUseCCTM else image, bbox, myTemplateManager, image.shape)
        if smallestScore > scoreThreshold:
            continue
        bestTemplate = copy.deepcopy(myTemplateManager.GetTemplate(indexSmallestShapeDifference))
        newbbox, additionalScaleFactorForTemplate = ResizeBBoxAccordingToTemplate(bbox, bestTemplate)
        bestTemplate.RescaleThisTemplate(bestTemplate.scaleFactor * additionalScaleFactorForTemplate)
        templateH, templateW = bestTemplate.templateH, bestTemplate.templateW
        minY = round(newbbox[1] - templateH * 0.5)
        maxY = minY + templateH
        minX = round(newbbox[0] - templateW * 0.5)
        maxX = minX + templateW
        templatePatch = bestTemplate._image
        if maxY > imageHeight:
            templatePatch = templatePatch[0:(imageHeight - minY)]
            maxY = imageHeight
        if minY < 0:
            templatePatch = templatePatch[(0 - minY):, :]
            minY = 0
        if maxX > imageWidth:
            templatePatch = templatePatch[:, 0:(imageWidth - minX)]
            maxX = imageWidth
        if minX < 0:
            templatePatch = templatePatch[:, (0 - minX):]
            minX = 0
        imageForDebug[minY:maxY, minX:maxX] = templatePatch
        debugInfo['bestTemplateDebug'] = cv2.resize(debugInfo['bestTemplateDebug'], templatePatch.shape[::-1])
        try:
            if (maxX + templateW) < imageWidth:
                imageForDebug2[minY:maxY, minX + templateW:maxX + templateW] = debugInfo['bestTemplateDebug']
            else:
                imageForDebug2[minY:maxY, minX - templateW:maxX - templateW] = debugInfo['bestTemplateDebug']
        except:
            from IPython import embed; print('here!'); embed()
        # put score on image:
        cv2.putText(imageForDebug,
                text=str(smallestScore),
                org=(minX - 60, minY - 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=(0,),
                thickness=2,
                lineType=cv2.LINE_4)
        print("------ detection {} finish, linemode score {} ----------".format(indexDetection, smallestScore))
        if detectionOrigin[-1] == "\n":
            templatedDetections.append(detectionOrigin[:-1] + " " + str(indexSmallestShapeDifference) + "\n")
        else:
            templatedDetections.append(detectionOrigin + " " + str(indexSmallestShapeDifference))

        # write template index to yolo results
        pathTemplateIndexWithYoloResults = os.path.join(debugDir, '..', 'detectionID0', imageName + '.txt')
        if os.path.isfile(pathTemplateIndexWithYoloResults):
            with open(pathTemplateIndexWithYoloResults, 'a') as f:
                detection.append(str(indexSmallestShapeDifference))
                if isNewLineAtEnd:
                    detection[-1] = detection[-1] + '\n'
                f.write(' '.join(detection))
        else:
            with open(pathTemplateIndexWithYoloResults, 'w') as f:
                detection.append(str(indexSmallestShapeDifference))
                if isNewLineAtEnd:
                    detection[-1] = detection[-1] + '\n'
                f.write(' '.join(detection))

    if debugDir is not None:
        cv2.imwrite(os.path.join(debugDir, imageName + '.png'), imageForDebug)
        cv2.imwrite(os.path.join(debugDir, imageName + '_linemod.png'), imageForDebug2)
    return templatedDetections

def SaveSbn_ForLoopRoutine(i, myEventData, lastTimeStampRight):
    mysbn, lastTimeStamp = myEventData.StackOneFrame(EventsInETHZFormat.PopOneSbt, 720, 1280, timeLimit=0.02)
    print("left: {}".format(lastTimeStamp))
    # if (i % 10) != 0:
    #     continue
    cc = TransformSbn2Image(mysbn)
    
    cv2.imwrite(os.path.join('/home/runqiu/tmptmp/eventstereoslam-dataset/leftcam/', str(i).zfill(6) + '.png'), cc.astype('uint8'))
    mysbnRight, lastTimeStampRight = myEventData.StackOneFrame(EventsInETHZFormat.PopOneTimeLimitedSbn, 720, 1280, timeStampLimit=max(lastTimeStamp, lastTimeStampRight))
    print("right: {}".format(lastTimeStampRight))
    try:
        ccRight = TransformSbn2Image(mysbnRight)
    except:
        from IPython import embed; print('here!'); embed()
    cv2.imwrite(os.path.join('/home/runqiu/tmptmp/eventstereoslam-dataset/rightcam/', str(i).zfill(6) + '.png'), ccRight.astype('uint8'))


def DoEventlinemodDetection_ForLoopRoutine(myTemplateDetector, mysbn, frameCount):
    myTemplateDetector.DetectTemplatesSemiScaleInvariant(mysbn, minScale=0.4, maxScale=1.2, scaleMultiplier=1.2, debugPathRoot='/home/runqiu/tmptmp/debugEventLinemod/', dataSaveRoot='/media/runqiu/data/eventLinemodDatasets/validatorTraining', frameIndex=frameCount)
    frameCount += 1


if __name__ == "__main__":
    import argparse
    
    scriptName = os.path.basename(__file__)

    parser = argparse.ArgumentParser(
        description="Test detect object with event linemod lib",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Examples: \n" +
                '%s --templatepath .../templates/ --inputdata .../DVS_TEXT.txt --outputpath .../output/ \n' % scriptName
        )
    )

    parser.add_argument('--inputdata', '-i', action='store', type=str, dest='inputdata',
                        help='Path to the input data. Need to be ethz format (.txt)')
    parser.add_argument('--templatepath', '-t', action='store', type=str, dest='templatepath', default='./templates/',
                        help='Path to the templates. Need to contain all the images and json file with all the templateInfo. [default=%(default)s]')
    parser.add_argument('--outputpath', '-o', action='store', type=str, dest='outputpath', default='./output/',
                        help='Output path. [default=%(default)s]')
    parser.add_argument('--loglevel', '-l', action='store', type=str, dest='loglevel', default='DEBUG',
                        help='print log level. [default=%(default)s]')
    parser.add_argument('--yoloresultpath', '-y', action='store', type=str, dest='yoloresultpath', default='./',
                        help='path to the detection result from yolo. [default=%(default)s]')

    args, remaining = parser.parse_known_args()

    # make sure output path exists
    os.makedirs(args.outputpath, exist_ok=True)

    # config logger
    commonlogging.ConfigureRootLogger(os.path.join(args.outputpath, 'mylog.txt'), level=args.loglevel)

    # prepare templates
    templateInfos, templateData = ReadFromTemplateFolder(args.templatepath)
    myTemplateManager = EventLinemodTemplateManager(templateInfos, templateData)
    # myTemplateDetector = EventLinemodDetector(myTemplateManager, templateResponseThreshold=400)

    # start detection
    # myEventData = EventsInETHZFormat(args.inputdata)
    # myEventDataRight = EventsInETHZFormat(os.path.join(os.path.dirname(args.inputdata).rsplit('/', 1)[0], 'rightcam', 'DVS_TEXT.txt'))

    # load yolo detection results
    imageNames = glob.glob(os.path.join(args.yoloresultpath, 'leftcam', '*.png'))
    imageNames.sort()
    for indexImage, imageName in enumerate(imageNames):
        starttime = time.time()
        imageName = os.path.basename(imageName)
        if not (os.path.isfile(os.path.join(args.yoloresultpath, 'leftcam', 'labelsYolo', imageName.split('.png')[0] + '.txt')) and os.path.isfile(os.path.join(args.yoloresultpath, 'rightcam', 'labelsYolo', imageName.split('.png')[0] + '.txt'))):
            continue

        leftCamImage = cv2.imread(os.path.join(args.yoloresultpath, 'leftcam', imageName), cv2.IMREAD_GRAYSCALE)
        rightCamImage = cv2.imread(os.path.join(args.yoloresultpath, 'rightcam', imageName), cv2.IMREAD_GRAYSCALE)
        with open(os.path.join(args.yoloresultpath, 'leftcam', 'labelsYolo', imageName.split('.png')[0] + '.txt'), 'r') as f:
            detectionsLeftCam = f.readlines()
        with open(os.path.join(args.yoloresultpath, 'rightcam', 'labelsYolo', imageName.split('.png')[0] + '.txt'), 'r') as f:
            detectionsRightCam = f.readlines()

        detectionsWithTemplIDsLeftCam = DoTemplateRecognitionForDetections(detectionsLeftCam, leftCamImage, myTemplateManager, debugDir=os.path.join(args.yoloresultpath, 'leftcam', 'debug'), imageName=imageName.split('.png')[0])
        logger.debug("====right cam====")
        detectionsWithTemplIDsRIghtCam = DoTemplateRecognitionForDetections(detectionsRightCam, rightCamImage, myTemplateManager, debugDir=os.path.join(args.yoloresultpath, 'rightcam', 'debug'), imageName=imageName.split('.png')[0])
        with open(os.path.join(args.yoloresultpath, 'leftcam', 'labelsYoloTemplated', imageName.split('.png')[0] + '.txt'), 'w') as f:
            f.writelines(detectionsWithTemplIDsLeftCam)
        with open(os.path.join(args.yoloresultpath, 'rightcam', 'labelsYoloTemplated', imageName.split('.png')[0] + '.txt'), 'w') as f:
            f.writelines(detectionsWithTemplIDsRIghtCam)

        logger.debug("--------- finished one frame [%s] in %f seconds ----------", imageName, time.time() - starttime)

        
