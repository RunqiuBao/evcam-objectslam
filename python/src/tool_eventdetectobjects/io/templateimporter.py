import os
import json
import cv2

from tool_eventdetectobjects.commonutils.datatypeutils import MyJsonDecoder

import logging
logger = logging.getLogger(__name__)


def ReadFromTemplateFolder(templatePath):
    '''
    Args:
        templatePath: string. The path to a folder of templates, including *.png and templateInfos.json (camInObjectTransformation).
    '''
    with open(os.path.join(templatePath, 'templateInfos.json'), 'r') as jsonFile:
        templateInfos = json.load(jsonFile, cls=MyJsonDecoder)
    templateInfos = [templateInfo for templateInfo in templateInfos if 'templId' in templateInfo]
    logger.debug("found %d templates. importing...", len(templateInfos))
    templateData = []
    for templateInfosEntry in templateInfos:
        if 'templId' not in templateInfosEntry:
            continue
        templateData.append(cv2.imread(os.path.join(templatePath, str(templateInfosEntry['templId']).zfill(6) + '.png'), cv2.IMREAD_GRAYSCALE))

    return templateInfos, templateData