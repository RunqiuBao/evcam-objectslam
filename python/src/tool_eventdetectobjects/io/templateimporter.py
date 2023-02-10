import os
import json
import cv2

import logging
logger = logging.getLogger(__name__)


def ReadFromTemplateFolder(templatePath):
    '''
    Args:
        templatePath: string. The path to a folder of templates, including *.png and templateInfos.json (camInObjectTransformation).
    '''
    with open(os.path.join(templatePath, 'templateInfos.json'), 'r') as jsonFile:
        templateInfo = json.load(jsonFile)
    logger.debug("found %d templates. importing...", len(templateInfo))
    templateData = []
    for templateInfoEntry in templateInfo:
        templateData.append(cv2.imread(os.path.join(templateData, str(templateInfoEntry['templId']).zfill(6) + '.png'), cv2.IMREAD_GRAYSCALE))

    return templateInfo, templateData