import logging
import os

class MyCustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    green = "\x1b[32:20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    boldRed = "\x1b[31;1m"
    purple = "\x1b[35:1m"
    reset = "\x1b[0m"
    format = '%(asctime)s_[%(levelname)s]_%(pathname)s(ln:%(lineno)d): %(message)s'
    FORMATS = {
        logging.DEBUG: green + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: boldRed + format + reset
    }

    def format(self, record):
        logFmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(logFmt)
        return formatter.format(record)

def _GetCanonicalLogLevel(level):
    loggingLevel = {logging.CRITICAL, logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG}
    if level not in loggingLevel:
        level = {
            'CRITICAL': logging.CRITICAL,
            'ERROR': logging.ERROR,
            'WARNING': logging.WARNING,
            'INFO': logging.INFO,
            'DEBUG': logging.DEBUG
        }.get(('%s' % level).upper(), None)
    return level

def ConfigureRootLogger(logFilePath, level='DEBUG'):    
    root = logging.getLogger()
    level = _GetCanonicalLogLevel(level)

    root.setLevel(level)
    
    if os.path.exists(logFilePath):
        import subprocess
        from datetime import datetime
        archiveLogFilePath = logFilePath + '.{}'.format(str(datetime.now()).replace(' ', '_'))
        cmdString = "mv {} {}".format(logFilePath, archiveLogFilePath)
        subprocess.call(cmdString.split())
    file_handler = logging.FileHandler(logFilePath)
    # test.logに出力するログレベルを個別でERRORに設定
    file_handler.setLevel(level)
    file_handler.setFormatter(MyCustomFormatter())

    # コンソールに出力する
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(MyCustomFormatter())

    # loggerにそれぞれのハンドラーを追加
    root.addHandler(file_handler)
    root.addHandler(stream_handler)
