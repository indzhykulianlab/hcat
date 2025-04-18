from typing import *

import torch
from torch import Tensor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator

from hcat.backends.convnext import ConvNeXt
from hcat.state import Cell, Piece


def init_model():
    """
    Initalizes the Faster RCNN detection model for HCAT.

    :return: Faster RCNN model
    """
    backbone = ConvNeXt(
        in_channels=3,
        dims=[128, 256, 512, 1024],
        depths=[3, 3, 27, 3],
        out_channels=256,
    )
    backbone.out_channels = 256

    anchor_sizes = ((16,), (32,), (64,), (128,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    HairCellConvNext = FasterRCNN(
        backbone,
        num_classes=3,
        rpn_anchor_generator=anchor_generator,
        min_size=256,
        max_size=600,
    )

    return HairCellConvNext


def model_from_path(path: str, device: str):
    """
    Loads a FasterRCNN model from a url OR from a local source if available.

    :param url: URL of pretrained model path. Will save the model to the source directory of HCAT.
    :param device: Device to load the model to.
    :return:
    """
    model = init_model()
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    if "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    model = model.to(device)
    model.load_state_dict(checkpoint)

    return model.eval().to(memory_format=torch.channels_last)


def eval_crop(
    crop: Tensor,
    model: FasterRCNN,
    device: str,
    parent: Piece,
    coords: List[float] | None = None,
    return_window: bool = False,
    metadata: Dict[str, str] | None = None
) -> List[Cell]:
    """
    Evaluates a model on a crop of an image and returns ALL detections,
    including bad ones...
    Does NOT do a NMS pass. All thresholding should be done by the user defined GUI functions...

    :param crop:
    :type crop:
    :param device:
    :type device:
    :return:
    :rtype:
    """
    # preprocess crop
    crop = crop.to(device)
    crop = crop.permute(2, 0, 1).unsqueeze(0)
    crop = crop.div(255)
    crop = crop[:, 0:3, ...]

    assert crop.ndim == 4, f"Image Shape Incompatable: {crop.shape}"

    model = model.eval().to(device)
    out = model(crop)[0]

    boxes = out["boxes"].cpu()
    labels = out["labels"].cpu()
    scores = out["scores"].cpu()

    cells = []

    id = 0
    for b, l, s in zip(boxes, labels, scores):
        type = "OHC" if l == 1 else "IHC"
        x0, y0, x1, y1 = b
        box = [x0.item(), y0.item(), x1.item(), y1.item()]
        id += 1

        while (
            id in parent._candidate_children.keys()
        ):  # ensure id is not in the parent already...
            id += 1
        c = Cell(id=id, parent=parent, score=s.item(), type=type, bbox=box)
        c.set_creator('model_predicted')  # need to change!!!
        c.set_image_adjustments_at_creation(parent.adjustments)
        if 'creator' in metadata:
            c.set_creator(metadata['creator'])

        cells.append(c)

    x, y = coords
    for c in cells:
        x0, y0, x1, y1 = c.bbox
        scale = 289 / parent.pixel_size_xy
        c.bbox = [
            (x0 * scale) + y,
            (y0 * scale) + x,
            (x1 * scale) + y,
            (y1 * scale) + x,
        ]  # adjust bbox position, idk why axis gets transposed... weird!
    if return_window:
        return cells, [x, y]
    else:
        return cells
