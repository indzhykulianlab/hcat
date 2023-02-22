import torchvision
import hcat
import os.path
import re
import torch
import wget
from hcat.backends.convNeXt import ConvNeXt
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator

def init_model():
    """
    Initalizes the Faster RCNN detection model for HCAT.

    :return: Faster RCNN model
    """
    backbone = ConvNeXt(in_channels=3, dims=[128, 256, 512, 1024], depths=[3, 3, 27, 3], out_channels=256)
    backbone.out_channels = 256

    anchor_sizes = ((16,), (32,), (64,), (128,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes,
                                       aspect_ratios=aspect_ratios)

    HairCellConvNext = FasterRCNN(backbone,
                                  num_classes=3,
                                  rpn_anchor_generator=anchor_generator,
                                  min_size=256, max_size=600, )

    return HairCellConvNext


def FasterRCNN_from_url(url: str,  device: str, path: str = None):
    """
    Loads a FasterRCNN model from a url OR from a local source if available.

    :param url: URL of pretrained model path. Will save the model to the source directory of HCAT.
    :param device: Device to load the model to.
    :return:
    """

    if path is None:  # we dont have a user defined file...
        path = os.path.join(hcat.__path__[0], 'detection_trained_model.trch')
        if not os.path.exists(path):
            wget.download(url=url, out=path) # this will download the file...

    # Check if the path exists from the user...
    if not os.path.exists(path):
        raise RuntimeError(f'Could not locate the file at path: {path}')

    model = init_model()

    try:
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
    except:
        raise RuntimeError(f'Could not load torch model from file: {path}')

    if 'model_state_dict' in checkpoint:
        checkpoint = checkpoint['model_state_dict']
    model = model.to(device)
    model.load_state_dict(checkpoint)

    return model.eval().to(memory_format=torch.channels_last)


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
