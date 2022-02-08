import hcat.lib.functional
from hcat.lib.utils import calculate_indexes, load, cochlea_to_xml, correct_pixel_size
from hcat.lib.cell import Cell
from hcat.lib.cochlea import Cochlea
from hcat.backends.detection import FasterRCNN_from_url
from hcat.backends.detection import HairCellConvNext
from hcat.lib.utils import warn

import torch
from torch import Tensor
from torchvision.models.detection import FasterRCNN
from tqdm import tqdm
from itertools import product
import numpy as np
from hcat.lib.explore_lif import get_xml
import torchvision.ops
import skimage.io as io
from numpy.typing import NDArray

import os.path
from typing import Optional, List, Dict, Callable, Tuple

__model_url__ = 'https://github.com/buswinka/hcat/blob/master/modelfiles/detection.trch?raw=true'

@torch.no_grad()
def _detect(f: str,
            curve_path: str = None,
            cell_detection_threshold: float = 0.57,
            nms_threshold: float = 0.1,
            dtype: Optional[bool] = None,
            model: Optional[FasterRCNN] = None,
            scale: Optional[int] = None,
            save_xml: Optional[bool] = False,
            save_fig: Optional[bool] = False,
            pixel_size: Optional[bool] = None,
            cell_diameter: Optional[bool] = None):
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
        warn('ERROR: No File to Analyze... \nAborting.', color='red')
        return None
    if not pixel_size:
        warn('WARNING: Pixel Size is not set. Defaults to 288.88 nm x/y. '
             'Consider suplying value for optimal performance.', color='yellow')

    device = hcat.lib.utils.get_device(verbose=True)

    # Load and preprocess Image
    image_base: NDArray = load(f, 'TileScan 1 Merged', verbose=True)  # from hcat.lib.utils
    image_base: NDArray = image_base[[2, 3], ...].max(-1) if image_base.ndim == 4 else image_base

    dtype = image_base.dtype if dtype is None else dtype
    scale: int = scale if scale else hcat.lib.utils.get_dtype_offset(dtype)
    image_base: Tensor = torch.from_numpy(image_base.astype(np.uint16) / scale).to(device)
    image_base: Tensor = correct_pixel_size(image_base, current_pixel_size=pixel_size, cell_diameter=cell_diameter, verbose=True)
    _, x, y = image_base.shape
    image_base: Tensor = torch.cat((torch.zeros((1, x, y), device=device), image_base), dim=0)


    print(f'DONE: shape: {image_base.shape}, min: {image_base.min()},'
           f' max: {image_base.max()}, dtype: {image_base.dtype}')


    # Initalize the model...
    model: FasterRCNN = model.to(device) if model else FasterRCNN_from_url(url=__model_url__, model=HairCellConvNext, device=device)

    # Initalize curvature detection
    predict_curvature = hcat.lib.functional.PredictCurvature(erode=3)  # Callable[[Tensor], Tuple[Tensor, Tensor]]

    # Get the indicies for evaluating cropped regions
    x_ind: List[List[int]] = calculate_indexes(10, 235, x, x)  # [[0, 255], [30, 285], ...]
    y_ind: List[List[int]] = calculate_indexes(10, 235, y, y)  # [[0, 255], [30, 285], ...]
    total: int = len(x_ind) * len(y_ind)

    # Initalize other small things
    cell_id = 1
    cells = []
    add_cell: Callable = cells.append  # stupid but done for speed

    for x, y in tqdm(product(x_ind, y_ind), total=total, desc='Detecting: '):

        # Load and prepare image crop for ML model evaluation
        image: Tensor = image_base[:, x[0]:x[1], y[0]:y[1]]

        if image.max() == 0 or image.min() == image.max():
            continue

        # Evaluate Deep Learning Model
        out = model([image.float()])
        out = out[1][0] if isinstance(out, tuple) else out[0]

        ind: Tensor = out['scores'] > cell_detection_threshold

        scores: Tensor = out['scores'][ind].cpu()
        boxes: Tensor = out['boxes'][ind, :]
        labels: Tensor = out['labels'][ind]

        # The model output coords with respect to the crop of image_base. We have to adjust
        # Format of torch images expect to be [B, C, Height, Width]
        boxes[:, [0, 2]] += y[0]
        boxes[:, [1, 3]] += x[0]

        # center x, center y, width, height
        centers: Tensor = torchvision.ops.box_convert(boxes, 'xyxy', 'cxcywh').cpu()
        cx = centers[:, 0]
        cy = centers[:, 1]

        for i, score in enumerate(scores):
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
        warn('WARNING: All three methods to predict hair cell path have failed. Frequency Mapping functionality is '
             'limited. Consider Manual Calculation.', color='yellow')

    # curvature estimation really only works if there is a lot of tissue...
    if distance is not None and distance.max() > 4000:
        for c in cells: c.calculate_frequency(curvature[[0, 1], :], distance)  # calculate cell's best frequency
        cells = [c for c in cells if not c._distance_is_far_away]  # remove a cell if its far away from curve

    else:
        curvature, distance, apex = None, None, None
        warn('WARNING: Predicted Cochlear Distance is below 4000um. Not sufficient '
             'information to determine cell frequency.', color='yellow')

    xml = get_xml(f) if f.endswith('.lif') else None
    filename = os.path.split(f)[-1]

    # remove weird cell ID's
    for i, c in enumerate(cells): c.id = i+1

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

    c.write_csv()

    if save_xml: cochlea_to_xml(c)
    if save_fig: c.make_detect_fig(image_base)

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
