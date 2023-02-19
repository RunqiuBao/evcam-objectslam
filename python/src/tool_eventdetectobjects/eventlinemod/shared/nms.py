import numpy


def DoNonMaxSuppression(detections, boxes, scores, overlapThreshold=0.3, isMaxScoreBest=False):
    if len(detections) == 0:
        return []
    
    indexPickedBoxes = []
    pickedBoxes = []
    pickedDetections = []
    pickedScores = []

    if isinstance(boxes, list):
        boxes = numpy.array(boxes)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    indices = numpy.argsort(scores)
    if not isMaxScoreBest:
        indices = indices[::-1]

    while len(indices) > 0:
        last = len(indices) - 1
        i = indices[last]
        indexPickedBoxes.append(i)
        pickedDetections.append(detections[i])
        pickedBoxes.append(boxes[i])
        pickedScores.append(scores[i])

        xx1 = numpy.maximum(x1[i], x1[indices[:last]])
        yy1 = numpy.maximum(y1[i], y1[indices[:last]])
        xx2 = numpy.minimum(x2[i], x2[indices[:last]])
        yy2 = numpy.minimum(y2[i], y2[indices[:last]])

        width = numpy.maximum(0, xx2 - xx1 + 1)
        height = numpy.maximum(0, yy2 - yy1 + 1)

        overlap = (width * height) / area[indices[:last]]
        indices = numpy.delete(indices, numpy.concatenate(([last], numpy.where(overlap > overlapThreshold)[0])))

    return pickedDetections, pickedBoxes, pickedScores