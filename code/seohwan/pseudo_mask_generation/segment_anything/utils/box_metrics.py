def calculate_iou(box1, box2):
    # Calculate the intersection over union of two bounding boxes
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = intersection / float(area1 + area2 - intersection)
    return iou

def calculate_dice(box1, box2):
    # Calculate the Dice coefficient of two bounding boxes
    intersection = calculate_iou(box1, box2) * ((box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1))
    area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    dice = (2 * intersection) / (area1 + area2)
    return dice