import hcat.lib.functional
from hcat.lib.utils import calculate_indexes, load, cochlea_to_xml, correct_pixel_size_image, \
    rescale_box_sizes, normalize_image, make_rgb, image_to_float, save_image_as_png
from hcat.lib.cell import Cell
from hcat.lib.cochlea import Cochlea
from hcat.backends.detection import FasterRCNN_from_url
from hcat.lib.utils import warn
from torchvision.transforms.functional import gaussian_blur

import torch
from torch import Tensor
from torchvision.models.detection import FasterRCNN
from tqdm import tqdm
from itertools import product
import numpy as np
from hcat.lib.explore_lif import get_xml
import torchvision.ops

import os.path
from typing import Optional, List, Dict, Callable, Tuple, Union

__model_url__ = 'https://www.dropbox.com/s/opf43jwcbgz02vm/detection_trained_model.trch?dl=1'

@torch.no_grad()
def _detect(f: Optional[str] = None,
            image_base: Optional[Tensor] = None,
            curve_path: str = None,
            cell_detection_threshold: float = 0.8,
            nms_threshold: float = 0.3,
            dtype: Optional[bool] = None,
            model: Optional[FasterRCNN] = None,
            scale: Optional[int] = None,
            save_xml: Optional[bool] = False,
            save_fig: Optional[bool] = False,
            save_png: Optional[bool] = False,
            save_csv: Optional[bool] = True,
            normalize: Optional[bool] = False,
            pixel_size: Optional[Union[bool, int, float]] = None,
            cell_diameter: Optional[Union[bool, int, float]] = None,
            return_model: Optional[bool] = False,
            predict_curve: Optional[bool] = False,
            verbose: Optional[bool] = True):
    """
    HCAT detection algorithm.

    This algorithm will iterate over an arbirarily large image and excecute a pretrained Faster RCNN detection algorithm
    to detect and classify cochlear hair cells. Not meant to be called directly.

    :param f: file path to an image to be loaded and preprocessed. Image loading and preprocessing may be bypassed by passing image data with the kwarg "image_base".
    :param image_base:  An arbitrarily large torch.Tensor of a preprocessed image to analyze. May be ignored if a filepath to an image is passed via kwarg "f".
    :param curve_path: File path to a pre-calculated list of points denoting the curve of the cochlea from fiji.
    :param cell_detection_threshold: Rejection thresholds of cell predictions by likelihood score.
    :param nms_threshold: NMS intersection over union threshold for removing overlapping bounding boxes.
    :param dtype: Data type of input image. Valid options are ['uint8', 'uint16', 'uint12', 'float64']
    :param model: An optional pretrained Faster RCNN detection model. Will default to the pretrained Faster RCNN with a ConvNeXT backbone.
    :param scale: Optional dtype scaling factor by which to normalize image. Will be inferred from dtype if possible.
    :param save_xml: If true, HCAT will save the results of the analysis as bounding boxes in the PASCAL format.
    :param save_fig: If true, HCAT will save a rendering of the original image with a cell detections overlay.
    :param save_png: If true, HCAT will save a png of the input image.
    :param save_csv: If true, HCAT will save a csv of the cell detection analysis
    :param normalize: If true, HCAT will normalize the image pixels to lie between -1 and 1.
    :param pixel_size: X/Y pixel size (nm) of the input image. Will default to 288nm. Is mutually exclusive with cell_diameter.
    :param cell_diameter: The diameter of the cell in pixels. Will default to 30. Is mutually exclusive with pixel_size.
    :param return_model: Will return an instance of the Faster RCNN model used to detect hair cells.
    :param predict_curve: Will enable HCAT to perform the cochlea path estimation.
    :param verbose: Will print to standard out each step of HCAT

    :return: A cochlea object of the detection analysis.
    """

    if verbose: print(f'Performing Hair Cell Detection Task')
    if f is None and image_base is None:
        warn('ERROR: No File to Analyze... \nAborting.', color='red')
        return None
    if not (pixel_size or cell_diameter) and verbose:
        warn('WARNING: Pixel Size is not set. Defaults to 288.88 nm x/y. '
             'Consider supplying value for optimal performance.', color='yellow')

    device = hcat.lib.utils.get_device(verbose=verbose)

    if image_base is None and f:
        # Load and preprocess Image
        image_base: np.ndarray = load(f, 'reMerged', verbose=verbose)  # from hcat.lib.utils
        image_base: np.ndarray = image_base[:, ...].max(-1) if image_base.ndim == 4 else image_base

        dtype = image_base.dtype if dtype is None else dtype
        scale: int = scale if scale else hcat.lib.utils.get_dtype_offset(dtype, image_base.max())

        image_base: Tensor = image_to_float(image_base, scale, verbose=verbose)
        image_base: Tensor = image_base.to(device)
        image_base: Tensor = correct_pixel_size_image(image_base, pixel_size, cell_diameter, verbose=verbose)
        image_base: Tensor = make_rgb(image_base)
        image_base: Tensor = normalize_image(image_base, normalize, verbose=True)

    elif image_base is not None:
        image_base = image_base.to(device)

    if save_png:
        save_image_as_png(image_base, filename=f, verbose=verbose)


    # Initalize the model...
    if model is None:
        print(f'[      ] Initializing detection model...', end='')
        model: FasterRCNN = FasterRCNN_from_url(url=__model_url__, device=device)
        print("\r[\x1b[1;32;40m DONE \x1b[0m]")
    else:
        model = model.to(device)

    if device == 'mps':
        model = model.to(torch.float32)

    prepare_image: Callable = lambda x: x.to(torch.float32)  # x.sub(0.5).div(0.5).float()

    # Initalize curvature detection
    predict_curvature = hcat.lib.functional.PredictCurvature(erode=3)  # Callable[[Tensor], Tuple[Tensor, Tensor]]

    # Get the indicies for evaluating cropped regions
    _, x, y = image_base.shape
    if verbose: print(image_base.shape)
    x_ind: List[List[int]] = calculate_indexes(10, 235, x, x)  # [[0, 255], [30, 285], ...]
    y_ind: List[List[int]] = calculate_indexes(10, 235, y, y)  # [[0, 255], [30, 285], ...]
    total: int = len(x_ind) * len(y_ind)

    # Initalize other small things
    cell_id = 1
    cells = []
    add_cell: Callable = cells.append  # stupid but done for speed

    generator = tqdm(product(x_ind, y_ind), total=total, desc='Detecting: ') if verbose else product(x_ind, y_ind)

    for x, y in generator:

        # Load and prepare image crop for ML model evaluation
        image: Tensor = image_base[:, x[0]:x[1], y[0]:y[1]]

        if image.max() == 0 or image.min() == image.max():
            continue

        image: List[Tensor] = [prepare_image(image)]

        # Evaluate Deep Learning Model
        out = model(image)

        out = out[1][0] if isinstance(out, tuple) else out[0]

        ind: Tensor = out['scores'] > cell_detection_threshold

        scores: Tensor = out['scores'][ind].cpu()
        boxes: Tensor = out['boxes'][ind, :]
        labels: Tensor = out['labels'][ind]

        # The model output coords with respect to the crop of image_base. We have to adjust
        # Format of torch images expect to be [B, C, Height, Width]
        boxes[:, [0, 2]] += y[0]
        boxes[:, [1, 3]] += x[0]
        # boxes = rescale_box_sizes(boxes, current_pixel_size=pixel_size, cell_diameter=cell_diameter)

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
    if nms_threshold < 1:
        cells: List[Cell] = _cell_nms(cells, nms_threshold)

    ohc = sum([int(c.type == 'OHC') for c in cells])  # number of ohc
    ihc = sum([int(c.type == 'IHC') for c in cells])  # number of ihc
    if verbose:
        print(f'Total Cells: {len(cells)}\n   OHC: {ohc}\n   IHC: {ihc}')

    if predict_curve:
        max_projection: Tensor = image_base[[1], ...].mul(0.5).add(0.5).unsqueeze(-1).cpu()
        curvature, distance, apex = predict_curvature(max_projection, cells, curve_path)

        if curvature is None and verbose:
            warn('WARNING: All three methods to predict hair cell path have failed. Frequency Mapping functionality is '
                 'limited. Consider Manual Calculation.', color='yellow')

        # curvature estimation really only works if there is a lot of tissue...
        if distance is not None and distance.max() > 4000:
            for c in cells: c.calculate_frequency(curvature[[0, 1], :], distance)  # calculate cell's best frequency
            cells = [c for c in cells if not c._distance_is_far_away]  # remove a cell if its far away from curve

        else:
            curvature, distance, apex = None, None, None
            if verbose:
                warn('WARNING: Predicted Cochlear Distance is below 4000um. Not sufficient '
                     'information to determine cell frequency.', color='yellow')
    else:
        curvature, distance, apex = None, None, None

    xml = get_xml(f) if f and f.endswith('.lif') else None
    filename = os.path.split(f)[-1] if f else None

    # remove weird cell ID's
    for i, c in enumerate(cells): c.id = i + 1

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

    if save_csv: c.write_csv()
    if save_xml: cochlea_to_xml(c)
    if save_fig: c.make_detect_fig(image_base)

    if return_model:
        return c, model
    else:
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
