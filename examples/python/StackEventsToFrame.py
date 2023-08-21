# author: bao@robot.u-tokyo.ac.jp

import os
import sys
import cv2
import numpy

if os.getenv('RUNMODEPYTHON') is not None and os.getenv('RUNMODEPYTHON') == 'debug':
    # use local python packages for debug
    file_path = os.path.realpath(__file__)
    sys.path.append(os.path.join(file_path[:len(file_path) - len(file_path.split('/')[-1])], '../../', 'python/src/'))  # include local Python code.

from tool_eventdetectobjects.io.file_format.eventsinethzformat import EventsInETHZFormat
import tool_eventdetectobjects.logger.commonlogging as commonlogging

import logging
logger = logging.getLogger(__name__)


def TransformEventStackToImage(myEventStack):
    aa = numpy.abs(myEventStack)
    maxNonZero = cv2.findNonZero(aa)
    cc = myEventStack
    if maxNonZero is not None:
        bb = aa[maxNonZero[:, 0, 1], maxNonZero[:, 0, 0]]
        mildMaxValue = numpy.mean(bb) + numpy.std(bb) * 3
        cc = cc * 127 / mildMaxValue
    cc = numpy.clip(cc + 127, 0, 255).astype('uint8')
    return cc


def SaveEventStack_ForLoopRoutine(
    i,
    myEventData,
    myEventDataRight,
    lastTimeStampRight,
    datasetRootPath,
    imageWidth,
    imageHeight,
    stackingMode
):
    perStackTimeLength = 0.05
    if stackingMode == 'sbn':
        myStack, lastTimeStamp = myEventData.StackOneFrame(EventsInETHZFormat.PopOneTimeLimitedSbn, imageHeight, imageWidth, maxEventNumberPerFrame=40000, timeLimit=perStackTimeLength)
    elif stackingMode == 'sbt':
        myStack, lastTimeStamp = myEventData.StackOneFrame(EventsInETHZFormat.PopOneSbt, imageHeight, imageWidth, timeLimit=perStackTimeLength)
    else:
        logger.error("no such funcBreak")
        raise
    
    print("left: {}".format(lastTimeStamp))
    # if (i % 10) != 0:
    #     continue
    cc = TransformEventStackToImage(myStack)
    cv2.imwrite(os.path.join(datasetRootPath, 'leftcam', str(i).zfill(6) + '.png'), cc.astype('uint8'))
    if stackingMode == 'sbn':
        myStackRight, lastTimeStampRight = myEventDataRight.StackOneFrame(EventsInETHZFormat.PopOneSbt, imageHeight, imageWidth, timeStampLimit=max(lastTimeStamp, lastTimeStampRight))  # lastTimeStampRight might be larger than lastTimeStamp in case when there are only sparse events in one frame
    elif stackingMode == 'sbt':
        myStackRight, lastTimeStampRight = myEventDataRight.StackOneFrame(EventsInETHZFormat.PopOneSbt, imageHeight, imageWidth, timeStampLimit=max(lastTimeStamp, lastTimeStampRight))  # lastTimeStampRight might be larger than lastTimeStamp in case when there are only sparse events in one frame
    else:
        logger.error("no such funcBreak")
        raise
    print("right: {}".format(lastTimeStampRight))
    ccRight = TransformEventStackToImage(myStackRight)
    cv2.imwrite(os.path.join(datasetRootPath, 'rightcam/', str(i).zfill(6) + '.png'), ccRight.astype('uint8'))


if __name__ == "__main__":
    import argparse

    scriptname = os.path.basename(__file__)

    parser = argparse.ArgumentParser(
        description="Test stacking events to frame",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Examples: \n" +
                '%s --datasetRootPath .../dataset/ --inputdata .../DVS_TEXT.txt --loglevel DEBUG \n' % scriptname
        )
    )

    parser.add_argument('--inputdata', '-i', action='store', type=str, dest='inputdata',
                        help='Path to the input data. Need to be ethz format (.txt)')
    parser.add_argument('--datasetRootPath', '-o', action='store', type=str, dest='datasetRootPath',
                        help='Path to the dataset root')
    parser.add_argument('--loglevel', '-l', action='store', type=str, dest='loglevel', default='DEBUG',
                        help='print log level. [default=%(default)s]')

    args, remaining = parser.parse_known_args()

    # make sure output path exists
    os.makedirs(args.datasetRootPath, exist_ok=True)
    os.makedirs(os.path.join(args.datasetRootPath, 'leftcam'), exist_ok=True)
    os.makedirs(os.path.join(args.datasetRootPath, 'rightcam'), exist_ok=True)

    # config logger
    commonlogging.ConfigureRootLogger(os.path.join(args.datasetRootPath, 'mylog.txt'), level=args.loglevel)

    myEventData = EventsInETHZFormat(args.inputdata)
    myEventDataRight = EventsInETHZFormat(os.path.join(os.path.dirname(args.inputdata).rsplit('/', 1)[0], 'rightCam', 'DVS_TEXT.txt'))

    lastTimeStampRight = 0
    indexFrame = 0
    imageWidth = 1280
    imageHeight = 720
    while True:
        SaveEventStack_ForLoopRoutine(
            indexFrame,
            myEventData,
            myEventDataRight,
            lastTimeStampRight,
            args.datasetRootPath,
            imageWidth,
            imageHeight,
            'sbn'
        )
        indexFrame += 1
        logger.info("-------- one stack: %d--------", indexFrame)
