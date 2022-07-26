import torch
from torch import Tensor
import torchvision.ops
from typing import Union, Optional, Tuple, Dict

Device = Union[str, torch.device]


@torch.jit.script
def square_boxes(boxes: Tensor, size: int = 10) -> Tensor:
    boxes = torchvision.ops.box_convert(boxes, 'xyxy', 'cxcywh')
    boxes[:, [2, 3]] = size
    boxes = torchvision.ops.box_convert(boxes, 'cxcywh', 'xyxy')
    return boxes


@torch.jit.script
def calculate_accuracy(ground_truth: Dict[str, Tensor], predictions: Dict[str, Tensor],
                       device: Optional[Device] = None, threshold: float = 0.1) -> Tuple[Tensor]:

    device = device if device else ground_truth['boxes'].device

    # We want to set each box as a perfect square
    _gt = square_boxes(ground_truth['boxes'].to(device), size=10)
    _pred = square_boxes(predictions['boxes'].to(device), size=10)

    iou = torchvision.ops.box_iou(_gt, _pred)

    # GT: indices of GT boxes with CORRECT predictions
    gt_max, gt_indicies = iou.max(dim=1)
    gt = torch.logical_not(gt_max.gt(threshold)) if iou.shape[1] > 0 else torch.ones(0)

    # Pred: indices of prediction boxes with associated GT box
    pred = torch.logical_not(iou.max(dim=0)[0].gt(threshold)) if iou.shape[0] > 0 else torch.ones(0)

    true_positive = torch.sum(torch.logical_not(gt))
    false_positive = torch.sum(pred)
    false_negative = torch.sum(gt)

    tempmat = iou[gt.eq(0), :]
    if tempmat.numel() > 0:
        associated_box_ind = tempmat.argmax(1)
        # of all gt cells with a prediction, what percentage of them have the correct class label?
        cls_acc = (ground_truth['labels'][gt.eq(0)] == pred['labels'][associated_box_ind]).sum() / len(
            ground_truth['labels'][gt.eq(0)])
        ihc = (ground_truth['labels'][gt.eq(0)] == 1).int()
        ihc_acc = (ihc == pred['labels'][associated_box_ind]).sum() / ihc.gt(0).sum()
        ohc = (ground_truth['labels'][gt.eq(0)] == 2).int()
        ohc_acc = (ohc == pred['labels'][associated_box_ind]).sum() / ohc.gt(0).sum()
    else:
        ihc_acc = 0
        ohc_acc = 0
        cls_acc = 0


    return true_positive, false_positive, false_negative, cls_acc


