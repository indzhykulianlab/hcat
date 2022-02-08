import torchvision
import os.path
import re
import torch
import wget
import hcat
from hcat.backends.convNeXt import ConvNeXt

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator

# load a pre-trained model for classification and return
def _init_model():
    backbone = ConvNeXt(in_channels=3, dims=[128, 256, 512, 1024], depths=[3, 3, 27, 3], out_channels=256)
    path = '/media/DataStorage/Dropbox (Partners HealthCare)/HairCellInstance/models/convnext_base_22k_1k_384.pth'

    if os.path.exists(path):
        state_dict = torch.load(path)
        backbone.load_state_dict(state_dict['model'], strict=False)

    backbone.out_channels = 256

    backbone = torch.jit.script(backbone)

    anchor_sizes = ((16,), (32,), (64,), (128,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes,
                                       aspect_ratios=aspect_ratios)

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=7, sampling_ratio=2)

    HairCellConvNext = FasterRCNN(backbone,
                                  num_classes=3,
                                  rpn_anchor_generator=anchor_generator,
                                  # box_roi_pool=roi_pooler,
                                  min_size=256, max_size=600, )
    return HairCellConvNext


HairCellConvNext = _init_model()


def FasterRCNN_from_url(url: str, device: str, model: FasterRCNN = HairCellConvNext):
    """ loads model from url """
    path = os.path.join(hcat.__path__[0], 'detection.trch')

    if not os.path.exists(path):
        print('Downloading Model File: ')
        wget.download(url=url, out=path)
        print(' ')

    model = model.requires_grad_(False)
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    if 'model_state_dict' in checkpoint:
        checkpoint = checkpoint['model_state_dict']
    model = model.to(device)
    model.load_state_dict(checkpoint)

    return model.eval().to(memory_format=torch.channels_last)


def _is_url(input: str):
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    # Return true if its a url
    return re.match(regex, input) is not None
