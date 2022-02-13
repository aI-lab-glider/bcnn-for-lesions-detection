import cv2
import numpy as np


def is_contour_bad(c):
    _, radius = cv2.minEnclosingCircle(c)
    return radius > 10


def getXFromRect(item):
    return item[0]


def apply_variance_postprocess(variance: np.ndarray):
    lower_bound = np.percentile(variance, 99.6)
    variance[lower_bound > variance] = 0

    variance[np.nonzero(variance)] = 255
    variance = variance.astype(np.uint8)
    for i in range(variance.shape[0]):
        im = cv2.erode(variance[i], np.ones((3, 3)))
        variance[i, :, :] = cv2.dilate(im, np.ones((3, 3)))
    return variance
