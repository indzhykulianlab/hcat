import numpy as np


def histogram(image: np.array):
    """
    Calculates a reliable histogram of a uint8 or uint16 images

    :param image:
    :type image:
    :return:
    :rtype:
    """
    if image.dtype != np.uint8:
        image = (image / 255).round().clip(0, 255)

    position = np.arange(0, 256)
    hist = np.zeros_like(position)

    unique, counts = np.unique(image, return_counts=True)

    for i, u in enumerate(unique):
        hist[position == u] = counts[i]

    return hist, position
