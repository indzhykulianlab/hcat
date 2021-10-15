import hcat.lib.functional
from hcat.lib.utils import calculate_indexes, load, cochlea_to_xml, correct_pixel_size, scale_to_hair_cell_diameter
from hcat.lib.cell import Cell
from hcat.lib.cochlea import Cochlea
from hcat.backends.detection import FasterRCNN_from_url

import torch
from torch import Tensor
from tqdm import tqdm
from itertools import product
import numpy as np
from hcat.lib.explore_lif import get_xml
import torchvision.ops
import skimage.io as io

import os.path
from typing import List, Dict


# DOCUMENTED

def _detect(f: str, curve_path: str = None, cell_detection_threshold: float = 0.86, dtype=None,
            nms_threshold: float = 0.2, save_xml=False, save_fig=False, pixel_size=None, cell_diameter=None):
    """
    2D hair cell detection algorithm.
    Loads arbitrarily large 2d image and performs iterative faster rcnn detection on the entire image.

    :param *str* f: path to image by which to analyze
    :param *float* cell_detection_threshold: cells below threshold are rejected
    :param *float* nms_threshold: iou rejection threshold for nms.
    :return: *Cochlea* object containing data of analysis.
    """
    print('Initializing hair cell detection algorithm...')
    if f is None:
        print('\x1b[1;31;40m' + 'ERROR: No File to Analyze... \nAborting.' + '\x1b[0m')
        return None
    if not pixel_size:
        print('\x1b[1;33;40m'
              'WARNING: Pixel Size is not set. Defaults to 288.88 nm x/y. Consider suplying value for optimal performance.'
              '\x1b[0m')

    with torch.no_grad():

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            print('\x1b[1;32;40mCUDA: GPU successfully initialized!\x1b[0m')
        else:
            print('\x1b[1;33;40m'
                  'WARNING: GPU not present or CUDA is not correctly intialized for GPU accelerated computation. '
                  'Analysis may be slow.'
                  '\x1b[0m')

        # Load and preprocess Image
        image_base = load(f, 'TileScan 1 Merged', verbose=True)  # from hcat.lib.utils
        image_base = image_base[[2, 3],...].max(-1) if image_base.ndim == 4 else image_base

        shape = list(image_base.shape)
        shape[0] = 1

        dtype = image_base.dtype if dtype is None else dtype
        scale: int = hcat.lib.utils.get_dtype_offset(dtype)


        temp = np.zeros(shape)
        temp = np.concatenate((temp, image_base)) / scale * 255
        io.imsave(f[:-4:]+'.png', temp.transpose((1,2,0)))

        c, x, y = image_base.shape
        print(
            f'DONE: shape: {image_base.shape}, min: {image_base.min()}, max: {image_base.max()}, dtype: {image_base.dtype}')

        if image_base.max() < scale * 0.33:
            print('\x1b[1;33;40m'
                  f'WARNING: Image max value less than 1/3 the scale factor for bit depth. Image Max: {image_base.max()},'
                  f' Scale Factor: {scale}, dtype: {dtype}. Readjusting scale to 1.5 time Image max.'
                  '\x1b[0m')
            scale = image_base.max() * 1.5



        image_base = torch.from_numpy(image_base.astype(np.uint16) / scale).to(device)

        if pixel_size is not None:
            image_base: Tensor = correct_pixel_size(image_base, pixel_size) #model expects pixel size of 288.88
            print(f'Rescaled Image to match pixel size of 288.88nm with a new shape of: {image_base.shape}')

        elif cell_diameter is not None:
            image_base: Tensor = scale_to_hair_cell_diameter(image_base, cell_diameter)
            print(f'Rescaled Image to match pixel size of 288.88nm with a new shape of: {image_base.shape}')

        # normalize around zero
        image_base.sub_(0.5).div_(0.5)



        # Initalize the model...
        model = FasterRCNN_from_url(url='https://github.com/buswinka/hcat/blob/master/modelfiles/detection.trch?raw=true', device=device)
        model.eval()


        # Initalize curvature detection
        predict_curvature = hcat.lib.functional.PredictCurvature(erode=3)

        # Get the indicies for evaluating cropped regions
        c, x, y = image_base.shape
        image_base = torch.cat((torch.zeros((1, x, y), device=device), image_base), dim=0)
        x_ind: List[List[int]] = calculate_indexes(10, 235, x, x)  # [[0, 255], [30, 285], ...]
        y_ind: List[List[int]] = calculate_indexes(10, 235, y, y)  # [[0, 255], [30, 285], ...]
        total: int = len(x_ind) * len(y_ind)

        # Initalize other small things
        cell_id = 1
        cells = []
        add_cell = cells.append  # stupid but done for speed

        for x, y in tqdm(product(x_ind, y_ind), total=total, desc='Detecting: '):

            # Load and prepare image crop for ML model evaluation
            image: Tensor = image_base[:, x[0]:x[1], y[0]:y[1]].unsqueeze(0)

            # If the image has nothing in it we can skip for speed
            if image.max() == -1:
                continue

            # Evaluate Deep Learning Model
            out: Dict[str, Tensor] = model(image.float())[0]

            scores: Tensor = out['scores'].cpu()
            boxes: Tensor = out['boxes'].cpu()
            labels: Tensor = out['labels'].cpu()

            # The model output coords with respect to the crop of image_base. We have to adjust
            # idk why the y and x are flipped. Breaks otherwise.
            boxes[:, [0, 2]] += y[0]
            boxes[:, [1, 3]] += x[0]

            # center x, center y, width, height
            centers: Tensor = torchvision.ops.box_convert(boxes, 'xyxy', 'cxcywh').cpu()
            cx = centers[:, 0]
            cy = centers[:, 1]

            for i, score in enumerate(scores):
                if score > cell_detection_threshold:
                    add_cell(Cell(id=cell_id,
                                  loc=torch.tensor([0, cx[i], cy[i], 0]),
                                  image=None,
                                  mask=None,
                                  cell_type='OHC' if labels[i] == 1 else 'IHC',
                                  boxes=boxes[i, :],
                                  scores=scores[i]))
                cell_id += 1

        # some cells may overlap. We remove cells after analysis is complete.
        cells: List[Cell] = _cell_nms(cells, nms_threshold)

        ohc = sum([int(c.type == 'OHC') for c in cells])  # number of ohc
        ihc = sum([int(c.type == 'IHC') for c in cells])  # number of ihc
        print(f'Total Cells: {len(cells)}\n   OHC: {ohc}\n   IHC: {ihc}' )

        max_projection: Tensor = image_base[[1], ...].mul(0.5).add(0.5).unsqueeze(-1).cpu()
        curvature, distance, apex = predict_curvature(max_projection, cells, curve_path)

        if curvature is None:
            print('\x1b[1;33;40mWARNING: ' +
                  'All three methods to predict hair cell path have failed. Frequency Mapping functionality is limited.'
                  'Consider Manual Calculation.'
                  + '\x1b[0m')

        # curvature estimation really only works if there is a lot of tissue...
        if distance is not None and distance.max() > 4000:
            for c in cells: c.calculate_frequency(curvature[[0, 1], :], distance)  # calculate cell's best frequency
            cells = [c for c in cells if not c._distance_is_far_away]  # remove a cell if its far away from curve

        else:
            curvature, distance, apex = None, None, None
            print('\x1b[1;33;40mWARNING: ' +
                  'Predicted Cochlear Distance is below 4000um. Not sufficient information to determine cell frequency.'
                  + '\x1b[0m')

        xml = get_xml(f) if f.endswith('.lif') else None
        filename = os.path.split(f)[-1]

        # Store in compressible object for further use
        c = Cochlea(mask=None,
                    filename=filename,
                    path=f,
                    analysis_type='detect',
                    leica_metadata=xml,
                    im_shape=image_base.shape,
                    cochlear_distance=distance,
                    curvature=curvature,
                    cells=cells,
                    apex=apex)

        if save_xml: cochlea_to_xml(c)

        if save_fig: c.make_detect_fig(image_base)
        # c.make_cochleogram()
        print('')
        return c


def _cell_nms(cells: List[Cell], nms_threshold: float) -> List[Cell]:
    """
    Perforns non maximum supression on the resulting cell predictions

    :param cells: Iterable of cells
    :param nms_threshold: cell iou threshold
    :return: Iterable of cells
    """
    # nms to get rid of cells
    boxes = torch.zeros((len(cells), 4))
    scores = torch.zeros(len(cells))
    for i, c in enumerate(cells):
        boxes[i, :] = c.boxes
        scores[i] = c.scores

    ind = torchvision.ops.nms(boxes, scores, nms_threshold)

    # need to pop off list elements from an int64 tensor
    ind_bool = torch.zeros(len(cells))
    ind_bool[ind] = 1
    for i, val in enumerate(ind_bool):
        if val == 0:
            cells[i] = None

    return [c for c in cells if c]
