# import tool_eventdetectobjects.logger.commonlogging as commonlogging
# author: bao@robot.u-tokyo.ac.jp

import sys
import os
import cv2
import time
import numpy

import tool_eventdetectobjects.logger.commonlogging as commonlogging
from tool_eventdetectobjects.io.file_format.eventsinethzformat import EventsInETHZFormat
from tool_eventdetectobjects.io.templateimporter import ReadFromTemplateFolder
from tool_eventdetectobjects.eventlinemod.shared.eventlinemodtemplatemanager import EventLinemodTemplateManager
from tool_eventdetectobjects.eventlinemod.eventlinemoddetector import EventLinemodDetector

import logging
logger = logging.getLogger(__name__)


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

    args, remaining = parser.parse_known_args()

    # make sure output path exists
    os.makedirs(args.outputpath, exist_ok=True)

    # config logger
    commonlogging.ConfigureRootLogger(os.path.join(args.outputpath, 'mylog.txt'), level=args.loglevel)

    # prepare templates
    templateInfos, templateData = ReadFromTemplateFolder(args.templatepath)
    myTemplateManager = EventLinemodTemplateManager(templateInfos, templateData)
    myTemplateDetector = EventLinemodDetector(myTemplateManager, templateResponseThreshold=350)

    # start detection
    myEventData = EventsInETHZFormat(args.inputdata)
    frameCount = 0
    for i in range(1000000):
        mysbn = myEventData.PopOneTimeLimitedSbn(20000, 720, 1280)
        aa = numpy.abs(mysbn)
        maxNonZero = cv2.findNonZero(aa)
        bb = aa[maxNonZero[:, 0, 1], maxNonZero[:, 0, 0]]
        mildMaxValue = numpy.mean(bb) + numpy.std(bb) * 3
        cc = mysbn * 127 / mildMaxValue
        cc = numpy.clip(cc + 127, 0, 255).astype('uint8')

        cv2.imwrite(os.path.join('/home/runqiu/tmptmp/eventobjectdetect-stereo-dataset/leftcam', str(i).zfill(6) + '.png'), cc.astype('uint8'))
        # myTemplateDetector.DetectTemplatesSemiScaleInvariant(mysbn, minScale=0.2, maxScale=0.42, scaleMultiplier=1.2, debugPathRoot='/home/runqiu/tmptmp/debugEventLinemod/', dataSaveRoot='/media/runqiu/data/eventLinemodDatasets/validatorTraining', frameIndex=frameCount)
        # frameCount += 1
        # if frameCount >= 20:
        #     break
