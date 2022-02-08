import torch
import torchvision.transforms.functional
import torch.nn.functional as F
from kornia.augmentation import RandomAffine3D
from PIL.Image import Image
import numpy as np
from typing import Dict, Tuple, Union, Sequence, List
import elasticdeform
import skimage.io as io
from hcat import ShapeError
import matplotlib.pyplot as plt
from torch import Tensor
import torchvision.transforms.functional as ttf

"""
dl = detection_dataset(path=['data/detection/train', './data/detection/train_external'],
                                  batch_size=32,
                                  simple_class=True,
                                  transforms=torchvision.transforms.Compose([
                                      t.random_crop(shape=(300, 300, 16), box_window=True),
                                      t.to_cuda(),
                                      t.affine3d(rate=0.33, scale=(0.85, 1.2)),  # broken 2d
                                      t.elastic_deformation(grid_shape=(15, 15), scale=1, rate=0.2),  # broken 2d
                                      t.random_h_flip(rate=0.5),
                                      t.random_v_flip(rate=0.5),
                                      t.gaussian_blur(rate=0.1, kernel_targets=torch.tensor([3])),
                                      t.adjust_brightness(range_brightness=(-0.1, 0.1), rate=0.33),
                                      t.random_noise(gamma=0.05, rate=0.2),
                                  ]))
"""


@torch.jit.script
def _get_box(mask: Tensor, device: str, threshold: int) -> Tuple[Tensor, Tensor]:
    # mask in shape of 300, 400, 1 [H, W, z=1]
    nonzero = torch.nonzero(mask)  # Q, 3=[x,y,z]
    label = mask.max()

    box = torch.tensor([-1, -1, -1, -1], dtype=torch.long, device=device)

    # Recall, image in shape of [C, H, W]

    if nonzero.numel() > threshold:
        x0 = torch.min(nonzero[:, 1])
        x1 = torch.max(nonzero[:, 1])
        y0 = torch.min(nonzero[:, 0])
        y1 = torch.max(nonzero[:, 0])

        if (x1 - x0 > 0) and (y1 - y0 > 0):
            box[0] = x0
            box[1] = y0
            box[2] = x1
            box[3] = y1

    return label, box


@torch.jit.script
def _remove_bad_masks(image: torch.Tensor, min_cell_volume: int, device: str) -> torch.Tensor:
    """
    Input an boolean image volume of shape [X, Y, Z] and determine the number of nonzero pixels.

    Sets the segmentation mask of a cell to ZERO if both conditions are met:
        1 - The cell is touching an edge (All edges of volume are considered)
        2 - The cell is smaller than a lower bound set by arg: min_cell_volume

    :param image: [X, Y, Z]
    :param min_cell_volume: min number of voxels needed to keep a cell mask as valid

    :return: torch.Tensor of shape [X, Y, Z]. Returns image if no cells are touching the edge.
    """
    ind = torch.nonzero(image)  # Determine location of nonzero values -> torch.tensor([:, [X, Y, Z]])

    # Loop over each edge (i) and check if the mask is touching.
    # Implicitly, if a nonzero value is zero it is on an edge.
    for i in range(3):
        remove_bool = torch.any(ind[:, i] == 0) or torch.any(ind[:, i] == image.shape[i] - 1)
        remove_bool = remove_bool if image.nonzero().shape[0] < min_cell_volume else False

        # Remove cell if it touches the edge and is small. No need for further computation.
        if remove_bool:
            image = torch.zeros(image.shape, device=device)
            break

    return image

@torch.jit.script
def merged_transform_2D(data_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    with torch.no_grad():
        #### CONSTANTS ####
        DEVICE: str = str(data_dict['image'].device)

        # Image should be in shape of [C, H, W]
        CROP_WIDTH = torch.tensor([300], device=DEVICE)
        CROP_HEIGHT = torch.tensor([300], device=DEVICE)

        AFFINE_RATE = torch.tensor(0.5, device=DEVICE)
        AFFINE_SCALE = torch.tensor((0.75, 1.25), device=DEVICE)
        AFFINE_YAW = torch.tensor((-180, 180), device=DEVICE)
        AFFINE_SHEAR = torch.tensor((-4, 4), device=DEVICE)

        FLIP_RATE = torch.tensor(0.5, device=DEVICE)

        BLUR_RATE = torch.tensor(0.3, device=DEVICE)
        BLUR_KERNEL_TARGETS = torch.tensor([3], device=DEVICE, dtype=torch.int)

        BRIGHTNESS_RATE = torch.tensor(0.5, device=DEVICE)
        BRIGHTNESS_RANGE_GFP = torch.tensor((-0.3, 0.05), device=DEVICE)
        BRIGHTNESS_RANGE_ACTIN = torch.tensor((-0.1, 0.125), device=DEVICE)

        CONTRAST_RATE = torch.tensor(0.3, device=DEVICE)
        CONTRAST_RANGE = torch.tensor((0.75, 2.), device=DEVICE)

        NOISE_GAMMA = torch.tensor(0.07, device=DEVICE)
        NOISE_RATE = torch.tensor(0.5, device=DEVICE)

        image: Tensor = data_dict['image'].to(DEVICE)
        boxes: Tensor = data_dict['boxes'].to(DEVICE)
        labels = data_dict['labels'].to(DEVICE)

        ## ---------- Random Crop ---------- ##
        w = CROP_WIDTH
        h = CROP_HEIGHT

        ind = torch.randint(boxes.shape[0], (1, 1), dtype=torch.long, device=DEVICE)  # randomly select box

        box = boxes[ind, :].squeeze()
        x0 = box[0]
        y0 = box[1]

        x0 = x0.sub(torch.floor(w / 2)).long()
        y0 = y0.sub(torch.floor(h / 2)).long()
        x1 = x0 + w
        y1 = y0 + h

        image = image[:, y0.item():y1.item(), x0.item():x1.item(), :]

        # Image should be of shape [C, H, W]

        ind_x = torch.logical_and(boxes[:, 0] < x1, boxes[:, 2] > x0)
        ind_y = torch.logical_and(boxes[:, 1] < y1, boxes[:, 3] > y0)
        ind = torch.logical_and(ind_x, ind_y)
        boxes = boxes[ind, :]

        labels = labels[ind]

        boxes[:, 0] -= x0
        boxes[:, 1] -= y0
        boxes[:, 2] -= x0
        boxes[:, 3] -= y0

        boxes[:, 0] = torch.clamp(boxes[:, 0], 0, w.item())
        boxes[:, 1] = torch.clamp(boxes[:, 1], 0, h.item())
        boxes[:, 2] = torch.clamp(boxes[:, 2], 0, w.item())
        boxes[:, 3] = torch.clamp(boxes[:, 3], 0, h.item())

        ## --------------- box 2 mask
        _, x, y, z = image.shape
        n = boxes.shape[0]

        mask = torch.zeros((n, x, y, z), device=DEVICE, dtype=torch.int)
        for i in range(n):
            x0 = boxes[i, 0]
            y0 = boxes[i, 1]
            x1 = boxes[i, 2]
            y1 = boxes[i, 3]

            mask[i, y0:y1, x0:x1, :] = labels[i]  # class label

        ## -------------------affine
        if torch.rand(1, device=DEVICE) < AFFINE_RATE:
            angle = (AFFINE_YAW[0] - AFFINE_YAW[1]) * torch.rand(1, device=DEVICE) + AFFINE_YAW[0]
            shear = (AFFINE_SHEAR[0] - AFFINE_SHEAR[1]) * torch.rand(1, device=DEVICE) + AFFINE_SHEAR[0]
            scale = (AFFINE_SCALE[0] - AFFINE_SCALE[1]) * torch.rand(1, device=DEVICE) + AFFINE_SCALE[0]

            image = ttf.affine(image[..., 0], angle=angle.item(), shear=[float(shear.item())], scale=scale.item(),
                               translate=[0, 0]).unsqueeze(-1)
            mask = ttf.affine(mask[..., 0], angle=angle.item(), shear=[float(shear.item())], scale=scale.item(),
                              translate=[0, 0]).unsqueeze(-1)

        ## ------------------- hflip
        if torch.rand(1, device=DEVICE) < FLIP_RATE:
            image = ttf.vflip(image.unsqueeze(1).transpose(1, -1).squeeze(-1))
            image = image.unsqueeze(-1).transpose(1, -1).squeeze(1)
            mask = ttf.vflip(mask.unsqueeze(1).transpose(1, -1).squeeze(-1))
            mask = mask.unsqueeze(-1).transpose(1, -1).squeeze(1)

        ## ------------------- vflip
        if torch.rand(1, device=DEVICE) < FLIP_RATE:
            image = ttf.hflip(image.unsqueeze(1).transpose(1, -1).squeeze(-1))
            image = image.unsqueeze(-1).transpose(1, -1).squeeze(1)
            mask = ttf.hflip(mask.unsqueeze(1).transpose(1, -1).squeeze(-1))
            mask = mask.unsqueeze(-1).transpose(1, -1).squeeze(1)

        ## ------------------- blur
        if torch.rand(1, device=DEVICE) < BLUR_RATE:
            kern: int = int(BLUR_KERNEL_TARGETS[int(torch.randint(0, len(BLUR_KERNEL_TARGETS), (1, 1)).item())].item())
            image = ttf.gaussian_blur(image.unsqueeze(1).transpose(1, -1).squeeze(-1), [kern, kern])
            image = image.unsqueeze(-1).transpose(1, -1).squeeze(1)

        ## ------------------- bright
        if torch.rand(1, device=DEVICE) < BRIGHTNESS_RATE:
            # Get random brightness value for actin
            actin = (BRIGHTNESS_RANGE_ACTIN[1] - BRIGHTNESS_RANGE_ACTIN[0]) * torch.rand((1), device=DEVICE) + \
                    BRIGHTNESS_RANGE_ACTIN[0]

            # Get random brightness value for gfp
            gfp = (BRIGHTNESS_RANGE_GFP[1] - BRIGHTNESS_RANGE_GFP[0]) * torch.rand((1), device=DEVICE) + \
                  BRIGHTNESS_RANGE_GFP[0]
            val = torch.cat((gfp, actin))
            val = val.reshape(image.shape[0], 1, 1, 1)
            image.add_(val)
            # image = torch.clamp(image, 0, 1)

        # ## ------------------- Contrast
        #
        if torch.rand(1, device=DEVICE) < CONTRAST_RATE or True:
            contrast_val = (CONTRAST_RANGE[1] - CONTRAST_RANGE[0]) * torch.rand((2), device=DEVICE) + \
                    CONTRAST_RANGE[0]

            for c in range(image.shape[0]):
                image[c, ..., 0] = ttf.adjust_contrast(image[[c], ..., 0], contrast_val[c].item()).squeeze(0)

        ## ------------------- noise
        if torch.rand(1, device=DEVICE) < NOISE_RATE:
            image.add_(torch.rand(image.shape, device=DEVICE) * NOISE_GAMMA)
            image = torch.clamp(image, 0, 1)

        ## ------------------- Corrections
        if mask.shape[0] == 0: print(f'{mask.shape}')
        processed_masks: List[torch.jit.Future[Tensor]] = []

        for i in range(mask.shape[0]):
            processed_masks.append(torch.jit.fork(_remove_bad_masks, mask[i, ...], 10, DEVICE))

        results: List[Tensor] = []
        for future in processed_masks:
            results.append(torch.jit.wait(future))
        mask = torch.stack(results)

        ## ------------------- Mask to Box

        _, x, y, z = image.shape
        n = mask.shape[0]

        boxes = torch.zeros([n, 4], device=DEVICE, dtype=torch.long)
        labels = torch.zeros([n], device=DEVICE, dtype=torch.long)

        processed_boxes: List[torch.jit.Future[Tuple[Tensor, Tensor]]] = []

        for i in range(n):
            # _get_box returns Tuple[Tensor (labels.shape=[N]), Tensor (boxes.shape=[N,4])]
            processed_boxes.append(torch.jit.fork(_get_box, mask=mask[i, ...], device=DEVICE, threshold=20))

        results: List[Tuple[Tensor, Tensor]] = []
        for future in processed_boxes:
            results.append(torch.jit.wait(future))

        for i in range(n):
            labels[i] = results[i][0]
            boxes[i, :] = results[i][1]

        ind = boxes.sum(dim=-1) != -4
        labels = labels[ind]
        boxes = boxes[ind, :]

        # ------------- WRAP UP

        data_dict['image'] = image
        data_dict['masks'] = torch.empty([0], device=DEVICE)
        data_dict['boxes'] = boxes
        data_dict['labels'] = labels

        return data_dict
