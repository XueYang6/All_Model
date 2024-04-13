import numpy as np
import cv2 as cv


def proba_metrics(proba, true):
    """

    :param proba:
    :param true:
    :param save_name:
    :return:
    """
    # Positions that are 1 in true are multiplied by proba
    intersection = np.sum(np.multiply(proba, true))
    # intersection add the proba value that is true to 0
    union = intersection + np.sum(proba * (1 - true))

    # calculate iou
    inf1 = intersection / union if union != 0 else 0

    # calculate dice
    inf2 = (2 * intersection) / (proba.sum() + true.sum()) if (proba.sum() + true.sum()) != 0 else 0

    return inf1, inf2


def euclidean_distance(point1: tuple, point2: tuple):
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]

    distance = ((dx ** 2) + (dy ** 2)) ** 0.5

    return dx, dy, distance


def seg_metrics(predict, true):
    intersection = np.sum(predict*true)
    union = np.logical_or(true, predict).sum()

    iou = intersection / union if union != 0 else 0

    dice = np.round(((2. * intersection) / (np.sum(true) + np.sum(predict))), 4)

    return iou, dice
