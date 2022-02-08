from __future__ import print_function
from __future__ import division
import torch
import torchvision


def rcnn(path=None):
    """
    Simple function to return faster rcnn model
    :param path: Path to pretrained model file.
    :return: faster_rcnn model
    """

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,
                                                                 progress=True,
                                                                 num_classes=3,
                                                                 pretrained_backbone=True,
                                                                 box_detections_per_img=500)
    if path is not None:
        model.load_state_dict(torch.load(path))
    return model

