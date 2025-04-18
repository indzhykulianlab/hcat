import os.path
import pickle
import re

import numpy as np
import skimage.io as io
import tqdm
import wget

import hcat
from hcat.state import Cochlea


def load_image(path: str) -> np.ndarray | None:
    """
    Loads an image from a file and ensures it is in the shape [C, X, Y, Z]

    :param path:
    :return:
    """
    if not os.path.exists(path):
        image = None
    else:
        image = io.imread(path)  # [X, Y], [X, Y, C], [Z, X, Y, C]
        # image = image if image.ndim != 2 else image[:, :, np.newaxis]

    if image.ndim == 2:
        image = image[:, :, np.newaxis, np.newaxis]
        image = image.transpose(-1, 1, 2, 0)

    elif image.ndim == 3: # Z or C ???
        image = image.transpose(-1, 0, 1)[..., np.newaxis]

    elif image.ndim == 4:
        image = image.transpose(-1, 1, 2, 0)

    return image

def imread(path: str) -> np.ndarray:
    _image = io.imread(path)
    if _image.max() > 256 and _image.max() < 2**12:
        _image = _image / 2 ** 12
        _image = _image * 255
        _image = np.round(_image).astype(np.uint8)
    elif _image.max() > 256 and _image.max() >= 2**12:
        _image = _image / 2 ** 16
        _image = _image * 255
        _image = np.round(_image).astype(np.uint8)


    if _image.shape[0] < 5:  # sometimes the channel is first i guess..
        _image = _image.transpose(1, 2, 0)

    while _image.shape[-1] < 3:
        x, y, _ = _image.shape
        _image = np.concatenate((_image, np.zeros((x, y, 1), dtype=np.uint8)), axis=-1)

    _image = np.ascontiguousarray(_image[:, :, 0:3].astype(np.uint8))
    return _image


def defualt_path_from_url(url: str):
    """
    Downloads a FasterRCNN model from a url if default model is not already available

    :param url: URL of pretrained model path. Will save the model to the source directory of HCAT.
    :param device: Device to load the model to.
    :return:
    """
    path = os.path.join(hcat.__path__[0], 'detection_trained_model.trch')
    print(path)
    if not os.path.exists(path) and _is_url(url):
        wget.download(url=url, out=path) # this will download the file...

    return path


def _is_url(input: str):
    """
    Checks if a string is a url.
    :param input: Any string
    :return: True if the string is a url.
    """
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    # Return true if its a url
    return re.match(regex, input) is not None


def save_state(state: Cochlea, filename: str) -> None:
    """
    Saves the cochlea object to a file via a pickle method

    :param state: Cochlea object
    :param filename: filename to save as
    :return: None
    """
    raise NotImplementedError

def load_state(filename: str) -> Cochlea | None:
    """ Loads the state from the filesystem """

    if os.path.exists(filename):
        return pickle.load(filename)
    else:
        return None

