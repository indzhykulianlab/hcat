from typing import Optional

import numpy as np
import skimage.io as io
import torch
from torch import Tensor


def imread(image_path: str,
           pin_memory: Optional[bool] = False) -> Tensor:
    """
    Imports an image from file and returns in torch format

    :param image_path: path to image
    :param pin_memory: saves torch tensor in pinned memory if true
    :return:
    """

    image: np.array = io.imread(image_path)  # [Z, X, Y, C]
    image: np.array = image[..., np.newaxis] if image.ndim == 3 else image
    image: np.array = image.transpose(-1, 1, 2, 0)
    image: np.array = image[[2], ...] if image.shape[0] > 3 else image  # [C=1, X, Y, Z]

    image: Tensor = torch.from_numpy(image)

    if pin_memory:
        image: Tensor = image.pin_memory()

    return image