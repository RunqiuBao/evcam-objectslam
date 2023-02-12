import numpy
import time
from os.path import exists, join
from os import makedirs

from commonutils.ioutils import LmdbWriter

import logging
logger = logging.getLogger(__name__)


class EventsInETHZFormat(object):
    _filePath = None
    _textFile = None
    _isEndOfFile = None

    def __init__(self, filePath):
        if exists(filePath):
            self._filePath = filePath
        else:
            raise FileNotFoundError("data file does not exit: %s", filePath)
        
        self._textFile = open(self._filePath, "r")
        lineString, self._textFile = self._ReadUntilFirstDataLine(self._textFile)  # throw away one line
        self._isEndOfFile = False

    def _ReadUntilFirstDataLine(self, _fileHandle):
        lineString = ""
        while lineString.startswith('#') or lineString == "":  # skip header of the file
            lineString = _fileHandle.readline()
        return lineString, _fileHandle

    def PopOneEvent(self):
        lineString = self._textFile.readline()
        lineList = lineString.split(' ')
        event = {'time': float(lineList[0]), 'x': int(lineList[1]), 'y': int(lineList[2]), 'polarity': int(lineList[3])}
        return event  # timestamp, x, y, polarity(0, 1)

    def PopOneTimeLimitedSbn(self, maxEventNumberPerFrame, imageSize, timeLimit=0.1):
        ts = []
        xs = []
        ys = []
        ps = []
        indexEvent = 0
        timeStampStart = None
        while indexEvent < maxEventNumberPerFrame:
            lineString = self._textFile.readline()
            if lineString == '':
                self._isEndOfFile = True
                break
            newevent = lineString.split(' ')  # (ts, x, y, p)
            if timeStampStart is None:
                timeStampStart = float(newevent[0])
            if (float(newevent[0]) - timeStampStart) >= timeLimit:
                logger.error("Event stacking has overpassed time limit %.6f sec.", timeLimit)
                break
            ts.append(float(newevent[0]))
            xs.append(int(newevent[1]))
            ys.append(int(newevent[2]))
            ps.append(int(newevent[3].split('\n')[0]))
            indexEvent += 1
        if self._isEndOfFile:
            logger.error("End of event data file.")
            return
        ts = numpy.array(ts)
        xs = numpy.array(xs)
        ys = numpy.array(ys)
        ps = numpy.array(ps)
        sbn = numpy.zeros(imageSize)
        psCentered = numpy.where(ps==0, -1, ps)
        numpy.add.at(sbn, (ys, xs), psCentered)
        return sbn        

    def ResetInputFile(self):
        self._textFile.seek(0, 0)

    def GenerateSbnIntoDb(self, lmdbOutputPath, seqNum, eventNumberPerFrame, imageSize, lengthOfSeq=None):
        frameNum = 0
        frameStack = []
        while not self._isEndOfFile:
            code = '%03d_%08d' % (int(seqNum), frameNum)
            code = code.encode()
            sbn = self.PopOneSbn(eventNumberPerFrame, imageSize)
            myresult = LmdbWriter.OneResult(code, sbn)
            frameStack.append(myresult)
            logger.debug('{}-th frame complete!'.format(frameNum))
            frameNum += 1
            if frameNum >= lengthOfSeq:
                break

        # write lmdb database
        logger.debug("-------start write database-------: {} frames in total, seq_num {}".format(frameNum, seqNum))
        starttime = time.time()
        makedirs(lmdbOutputPath, exist_ok=True)
        lmdbWriter = LmdbWriter(lmdbOutputPath)
        for ires, oneresult in enumerate(frameStack):
            lmdbWriter.write(oneresult.key, oneresult.img)
        lmdbWriter.commitchange()
        lmdbWriter.endwriting()
        logger.debug('-------finish write database-------: time cost {} s'.format(time.time() - starttime))

        # reset file
        self._textFile.seek(0, 0)


if __name__ == "__main__":
    dataRoot = '/home/runqiu/code/event_camera_repo/3rdparty/v2e/output/vibrationRollerNight/'
    myEventData = EventsInETHZFormat(join(dataRoot, 'DVS_TEXT.txt'))
    myEventData.GenerateSbnIntoDb(join(dataRoot, 'sbn'), 0, 20000, (720, 1280), lengthOfSeq=10)
