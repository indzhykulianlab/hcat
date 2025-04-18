from typing import Dict, Tuple, List, Callable

import torch
import torchvision.transforms.functional as ttf
from torch import Tensor
import torch.nn as nn
from yacs.config import CfgNode
import random
import logging



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


def _remove_bad_masks(image: Tensor, min_cell_volume: int, device: str) -> Tensor:
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


class TransformFromCfg(nn.Module):
    def __init__(self, cfg: CfgNode, device: torch.device, scale: float = 255.0):
        super(TransformFromCfg, self).__init__()
        """
        Why? Apparently a huge amount of overhead is just initializing this from cfg
        If we preinitalize, then we can save on overhead, to do this, we need a class...
        Probably a reasonalbe functional way to do this. Ill think on it later
        """

        self.prefix_function = self._identity
        self.posfix_function = self._identity

        self.dataset_mean = None
        self.dataset_std = None

        self.cfg = cfg

        self.DEVICE = device
        self.SCALE = scale

        self.CROP_WIDTH = cfg.AUGMENTATION.CROP_WIDTH
        self.CROP_HEIGHT = cfg.AUGMENTATION.CROP_HEIGHT

        self.CROP_DEPTH = cfg.AUGMENTATION.CROP_DEPTH

        self.FLIP_RATE = cfg.AUGMENTATION.FLIP_RATE

        self.BRIGHTNESS_RATE = cfg.AUGMENTATION.BRIGHTNESS_RATE
        self.BRIGHTNESS_RANGE = cfg.AUGMENTATION.BRIGHTNESS_RANGE
        self.NOISE_GAMMA = cfg.AUGMENTATION.NOISE_GAMMA
        self.NOISE_RATE = cfg.AUGMENTATION.NOISE_RATE

        self.FILTER_RATE = 0.5

        self.CONTRAST_RATE = cfg.AUGMENTATION.CONTRAST_RATE
        self.CONTRAST_RANGE = cfg.AUGMENTATION.CONTRAST_RANGE

        self.AFFINE_RATE = cfg.AUGMENTATION.AFFINE_RATE
        self.AFFINE_SCALE = cfg.AUGMENTATION.AFFINE_SCALE
        self.AFFINE_SHEAR = cfg.AUGMENTATION.AFFINE_SHEAR
        self.AFFINE_YAW = cfg.AUGMENTATION.AFFINE_YAW

    def _identity(self, *args):
        return args if len(args) > 1 else args[0]

    def _crop1(self, image, boxes, labels):
        w = self.CROP_WIDTH
        h = self.CROP_HEIGHT

        ind = torch.randint(boxes.shape[0], (1, 1), dtype=torch.long, device=self.DEVICE)  # randomly select box

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

        return image, boxes, labels


    def _flipX(self, image, boxes, labels):
        image = ttf.vflip(image.unsqueeze(1).transpose(1, -1).squeeze(-1))
        image = image.unsqueeze(-1).transpose(1, -1).squeeze(1)
        boxes[:, [3, 1]] = image.shape[2] - boxes[:, [1, 3]]
        return image,boxes, labels

    def _flipY(self, image, boxes,labels):
        image = ttf.hflip(image.unsqueeze(1).transpose(1, -1).squeeze(-1))
        image = image.unsqueeze(-1).transpose(1, -1).squeeze(1)
        boxes[:, [2, 0]] = image.shape[1] - boxes[:, [0, 2]]
        return image,boxes,labels


    def _invert(self, image, boxes, labels):
        image.sub_(255).mul_(-1)
        return image,boxes,labels

    def _brightness(self, image, boxes, labels):
        val = random.uniform(*self.BRIGHTNESS_RANGE)
        logging.debug(f"TransformFromCfg.forward() | adjusting brightness | {val=}")
        # in place ok because flip always returns a copy
        image = image.add(val)
        logging.debug(f'after brightness: {image.max()=}, {image.min()=}')
        image = image.clamp(0, 255)
        return image,boxes,labels

    def _contrast(self, image, boxes, labels):
        contrast_val = random.uniform(*self.CONTRAST_RANGE)
        logging.debug(f"TransformFromCfg.forward() | adjusting contrast | {contrast_val=}")
        # [ C, X, Y, Z ] -> [Z, C, X, Y]
        image = image.div(255)
        image = ttf.adjust_contrast(image.permute(3, 0, 1, 2), contrast_val).permute(1, 2, 3, 0)
        image = image.mul(255)
        logging.debug(f'after contrast: {image.max()=}, {image.min()=}')
        return image,boxes,labels


    def _noise(self, image, boxes, labels):
        noise = torch.rand(image.shape, device=self.DEVICE) * self.NOISE_GAMMA
        image = image.add(noise)
        return image,boxes,labels

    def _normalize(self, image, boxes, labels):
        # mean = image.float().mean()
        # std = image.float().std()
        mean = float(image.float().mean().cpu() if not self.dataset_mean else self.dataset_mean)
        std = float(image.float().std().cpu() if not self.dataset_std else self.dataset_std)

        image = image.float().sub(mean).div(std)
        return image,boxes,labels
    def set_dataset_mean(self, mean):
        self.dataset_mean = mean
        return self

    def set_dataset_std(self, std):
        self.dataset_std = std
        return self

    @torch.no_grad()
    def forward(self, data_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:

        assert "masks" in data_dict, 'keyword "masks" not in data_dict'
        assert "image" in data_dict, 'keyword "image" not in data_dict'

        data_dict = self.prefix_function(data_dict)
        logging.debug('augmenting the image... ')

        boxes= data_dict["boxes"]
        labels = data_dict['labels']
        image = data_dict["image"]
        logging.debug(f'pre-augmentation: {image.max()=}, {image.min()}')

        image, boxes, labels = self._crop1(image, boxes, labels)

        # ------------------- x flip
        if random.random() < self.FLIP_RATE:
            logging.debug("TransformFromCfg.forward() | flipping in x")
            image, boxes, labels = self._flipX(image, boxes, labels)

        # ------------------- y flip
        if random.random() < self.FLIP_RATE:
            logging.debug("TransformFromCfg.forward() | flipping in y")
            image, boxes, labels = self._flipY(image, boxes, labels)

        # ------------------- z flip
        if random.random() < self.FLIP_RATE:
            logging.debug("TransformFromCfg.forward() | flipping in z")
            image, boxes, labels = self._flipZ(image, boxes, labels)

        # # ------------------- Random Invert
        if random.random() < self.BRIGHTNESS_RATE:
            logging.debug("TransformFromCfg.forward() | inverting")
            image, boxes, labels = self._invert(image, boxes, labels)

        # ------------------- Adjust Brightness
        if random.random() < self.BRIGHTNESS_RATE:
            image, boxes, labels = self._brightness(image, boxes, labels)

        # ------------------- Adjust Contrast
        if random.random() < self.CONTRAST_RATE:
            logging.debug("TransformFromCfg.forward() | adjusting contrast")
            image, boxes, labels = self._contrast(image, boxes, labels)

        # ------------------- Noise
        if random.random() < self.NOISE_RATE:
            logging.debug("TransformFromCfg.forward() | adding noise")
            image, boxes, labels = self._noise(image, boxes, labels)

        image, boxes, labels = self._normalize(image, boxes, labels)

        data_dict["image"] = image
        data_dict["boxes"] =boxes
        data_dict['labels'] = labels

        data_dict = self.posfix_function(data_dict)

        return data_dict

    def pre_fn(self, fn: Callable[[Dict[str, Tensor]], Dict[str, Tensor]]):
        self.prefix_function = fn
        return self

    def post_fn(self, fn: Callable[[Dict[str, Tensor]], Dict[str, Tensor]]):
        self.posfix_function = fn
        return self

    def post_crop_fn(self, fn):
        self.postcrop_function = fn
        return self

    def __repr__(self):
        return f"TransformFromCfg[Device:{self.DEVICE}]\ncfg.AUGMENTATION:\n=================\n{self.cfg.AUGMENTATION}]"


@torch.no_grad()
def merged_transform_2D(data_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    # CONSTANTS #
    DEVICE: str = str(data_dict['image'].device)

    # Image should be in shape of [C, H, W]
    CROP_WIDTH = torch.tensor([300], device=DEVICE)
    CROP_HEIGHT = torch.tensor([300], device=DEVICE)

    AFFINE_RATE = torch.tensor(-1, device=DEVICE)
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

    # ---------- Random Crop
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

    # -------------------affine
    if torch.rand(1, device=DEVICE) < AFFINE_RATE:

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

        angle = (AFFINE_YAW[0] - AFFINE_YAW[1]) * torch.rand(1, device=DEVICE) + AFFINE_YAW[0]
        shear = (AFFINE_SHEAR[0] - AFFINE_SHEAR[1]) * torch.rand(1, device=DEVICE) + AFFINE_SHEAR[0]
        scale = (AFFINE_SCALE[0] - AFFINE_SCALE[1]) * torch.rand(1, device=DEVICE) + AFFINE_SCALE[0]

        image = ttf.affine(image[..., 0], angle=angle.item(), shear=[float(shear.item())], scale=scale.item(),
                           translate=[0, 0]).unsqueeze(-1)
        mask = ttf.affine(mask[..., 0], angle=angle.item(), shear=[float(shear.item())], scale=scale.item(),
                          translate=[0, 0]).unsqueeze(-1)

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

    # ------------------- horizontal flip
    if torch.rand(1, device=DEVICE) < FLIP_RATE:
        image = ttf.vflip(image.unsqueeze(1).transpose(1, -1).squeeze(-1))
        image = image.unsqueeze(-1).transpose(1, -1).squeeze(1)
        boxes[:, [3, 1]] = image.shape[2] - boxes[:, [1, 3]]

    # ------------------- vertical flip
    if torch.rand(1, device=DEVICE) < FLIP_RATE:
        image = ttf.hflip(image.unsqueeze(1).transpose(1, -1).squeeze(-1))
        image = image.unsqueeze(-1).transpose(1, -1).squeeze(1)
        boxes[:, [2, 0]] = image.shape[1] - boxes[:, [0, 2]]

    # ------------------- blur
    if torch.rand(1, device=DEVICE) < BLUR_RATE:
        kern: int = int(BLUR_KERNEL_TARGETS[int(torch.randint(0, len(BLUR_KERNEL_TARGETS), (1, 1)).item())].item())
        image = ttf.gaussian_blur(image.unsqueeze(1).transpose(1, -1).squeeze(-1), [kern, kern])
        image = image.unsqueeze(-1).transpose(1, -1).squeeze(1)

    # ------------------- bright
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

    # ------------------- Contrast
    if torch.rand(1, device=DEVICE) < CONTRAST_RATE or True:
        contrast_val = (CONTRAST_RANGE[1] - CONTRAST_RANGE[0]) * torch.rand((2), device=DEVICE) + \
                       CONTRAST_RANGE[0]

        for c in range(image.shape[0]):
            image[c, ..., 0] = ttf.adjust_contrast(image[[c], ..., 0], contrast_val[c].item()).squeeze(0)

    # ------------------- noise
    if torch.rand(1, device=DEVICE) < NOISE_RATE:
        image.add_(torch.rand(image.shape, device=DEVICE) * NOISE_GAMMA)
        image = torch.clamp(image, 0, 1)

    # ------------- WRAP UP
    ind = boxes[:, 0:2] < boxes[:, 2:]
    ind = torch.logical_and(ind[:, 0], ind[:, 1])

    boxes = boxes[ind, :]
    labels = labels[ind]

    data_dict['image'] = image
    data_dict['masks'] = torch.empty([0], device=DEVICE)
    data_dict['boxes'] = boxes
    data_dict['labels'] = labels

    return data_dict

