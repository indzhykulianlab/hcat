import hcat
import hcat.lib.utils as utils
import hcat.lib.functional as functional
from hcat.lib.utils import calculate_indexes, remove_edge_cells, remove_wrong_sized_cells, graceful_exit
from hcat.lib.cell import Cell
from hcat.lib.cochlea import Cochlea
from hcat.backends.spatial_embedding import SpatialEmbedding
from hcat.backends.unet_and_watershed import UNetWatershed

import torch
from torch import Tensor
from tqdm import tqdm
from itertools import product
import numpy as np
from hcat.lib.explore_lif import get_xml

import os.path
import gc
import psutil
from typing import Optional, List, Dict


# DOCUMENTED

def _segment(f: str, channel: int, intensity_reject_threshold: float,
             dtype: str = 'uint16',
             unet: bool = False, cellpose: bool = False, no_post: bool = False,
             figure: Optional[bool] = True) -> Cochlea:
    """

    Segment arbitrarily large images of cochleae.
    Loads a lif or tif file and iterativley segments all hair cells in the image.
    Saves the results of the analysis in a file with extension ".cochlea" which compresses and saves the analysis.

    Example:

    >>> from hcat.segment import _segment
    >>> f = 'path/to/my/file.tif'
    >>> segment(f, intensity_reject_threshold=0.5)

    :param *str* f: path to file
    :param *int* channel: index of cytosolic stain (0 if only one channel).
    :param *float* intensity_reject_threshold: predicted cell masks with mean cytosolic stain below this threshold will be rejected
    :param *str,optional* dtype: bit depth of input image. Possible values: uint8, uint16, int8, int16.
    :param *bool,optional* unet: Perform segmentation with UNet + Watershed segmentation backbone.
    :param *bool,optinal* cellpose: Perform segmentation with 3D cellpose segmentation backbone.
    :param *bool,optinal* no_post: if true, remove cell postprocessing, which may remove legitimate cells.
    :param *bool,optinal* figure: If true, renders a diagnostic figure for this cochlea.
    :return: None
    """
    print('Initializing hair cell segmentation algorithm...')
    if f is None:
        print('\x1b[1;31;40m' + 'ERROR: No File to Analyze... \nAborting.' + '\x1b[0m')
        return None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print('\x1b[1;32;40m' + 'CUDA: GPU successfully initialized!' + '\x1b[0m')
    else:
        print(
            '\x1b[1;33;40m' + 'WARNING: GPU not present or CUDA is not correctly intialized for GPU accelerated computation. '
                              'Analysis may be slow.' + '\x1b[0m')

    max_cell: int = 0
    cells = []
    append_cells = cells.append
    intensity_reject: callable = torch.jit.script(
        hcat.lib.functional.IntensityCellReject(threshold=intensity_reject_threshold).requires_grad_(False))
    predict_curvature: callable = hcat.lib.functional.PredictCurvature(erode=3)

    with torch.no_grad():

        backend = _get_backend(unet, cellpose, device)  # may return none if failed
        if backend is None:
            return None

        image_base = utils.load(f, 'TileScan 1 Merged', verbose=True)  # may return none if failed
        print('SEG SHAPE', image_base.shape)
        if image_base is None:
            return None

        print(
            f'DONE: shape: {image_base.shape}, min: {image_base.min()}, max: {image_base.max()}, dtype: {image_base.dtype}')

        dtype = image_base.dtype if dtype is None else dtype
        scale: int = hcat.lib.utils.get_dtype_offset(dtype)

        c, x, y, z = image_base.shape
        out_img = torch.zeros((1, x, y, z), dtype=torch.long)  # (1, X, Y, Z)

        x_ind: List[List[int]] = calculate_indexes(15, 255, x, x)  # [[0, 255], [30, 285], ...]
        y_ind: List[List[int]] = calculate_indexes(15, 255, y, y)  # [[0, 255], [30, 285], ...]
        total: int = len(x_ind) * len(y_ind)

        for (x, y) in tqdm(product(x_ind, y_ind), total=total, desc='Segmenting: '):

            # Load and prepare image for ML model evaluation
            image: np.array = image_base[:, x[0]:x[1], y[0]:y[1]].astype(np.float)
            image: Tensor = torch.from_numpy(image).float().div(scale).unsqueeze(0)
            image: Tensor = image.div(0.5) if image.min() < 0 else image.sub(0.5).div(0.5)  # signed or unsigned?

            # which backend do we use?
            out: Tensor = backend(image.to(device)) if unet else backend(image[:, [channel], ...].to(device))

            # backend will return an empty tensor if there wasn't usable data underneath
            if out.shape[1] == 0:
                continue

            # postprocess
            out: Tensor = intensity_reject(out, image[:, [channel], ...].to(device)) if not no_post else out

            # backend will return an empty tensor if there wasn't usable data underneath
            if out.shape[1] == 0:
                continue


            value, out = out.max(1)
            out[value < 0.25] = 0

            # postprocess
            out: Tensor = remove_wrong_sized_cells(out) if not no_post else out

            # Remove cells on the edge for proper merging
            out: Tensor = remove_edge_cells(out).cpu()

            if out.shape[1] == 0:
                continue

            out[out > 0] = out[out > 0] + max_cell
            max_cell = out.max() if out.max() > 0 else max_cell

            # merging!
            x_exp = torch.tensor([x[0], x[1]])
            y_exp = torch.tensor([y[0], y[1]])
            x_exp[x_exp < 0] = 0
            y_exp[y_exp < 0] = 0
            destination = out_img[:, x_exp[0]:x_exp[1], y_exp[0]:y_exp[1], :]
            destination, new_ids = functional.merge_regions(destination.to(out.device), out, 0.35)

            out_img[:, x_exp[0]:x_exp[1], y_exp[0]:y_exp[1], :] = destination.cpu()

            for u in new_ids:
                if u == 0:
                    continue
                center = (out == u).nonzero(as_tuple=False).float().mean(0)
                center[1] += x[0]
                center[2] += y[0]
                append_cells(Cell(id=u,
                                  image=torch.clone(image[:, :, :-1:, :-1:, 0:out.shape[-1]].mul(0.5).add(0.5)),
                                  mask=(out == u).unsqueeze(0),
                                  loc=center))
        print('Predicted Number of Cells: ', len(cells))

        mem = psutil.virtual_memory()  # get memory for later
        del image_base  # memory issues are tight. removing image and image_base help
        gc.collect()

        # Not every input will be a whole cochlea... curvature means nothing over such a small region
        curvature, distance, apex = predict_curvature(out_img.sum(-1).unsqueeze(-1), cells, None)

        if curvature is None:
            print('\x1b[1;33;40mWARNING: ' +
                  'All three methods to predict hair cell path have failed. Frequency Mapping functionality is limited.'
                  ' Consider Manual Calculation.'
                  + '\x1b[0m')
        # curvature estimation really only works if there is a lot of tissue...
        if distance is not None and distance.max() > 4000:
            for c in cells: c.calculate_frequency(curvature[[0, 1], :], distance)  # calculate cell's best frequency
            # cells = [c for c in cells if not c._distance_is_far_away]  # remove a cell if its far away from curve
        else:
            curvature, distance, apex = None, None, None
            print('\x1b[1;33;40mWARNING: ' +
                  'Predicted Cochlear Distance is below 4000um. Not sufficient information to determine cell frequency.'
                  + '\x1b[0m')

        xml = get_xml(f) if f.endswith('.lif') else None

        # Store
        c = Cochlea(mask=out_img,
                    filename=f,
                    leica_metadata=xml,
                    analysis_type='segment',
                    im_shape=out_img.shape,
                    cochlear_distance=distance,
                    curvature=curvature,
                    cells=cells,
                    apex=apex)

        # This takes too much memory for huge images
        # Check the size of the outimg. If we have twice the available free memory we can compress!
        if mem.available > (out_img.numpy().size * out_img.numpy().itemsize * 2):
            c.compress_mask(vebose=True)
        else:
            print('\x1b[1;33;40m' + 'WARNING: Low free system memory. Cannot compress output... ' + '\x1b[0m')

        c.save(os.path.splitext(f)[0] + '.cochlea')
        print('Saved: ', os.path.splitext(f)[0] + '.cochlea')

        if figure:
            c.make_fig(os.path.splitext(f)[0] + '.pdf')

        # c.write_csv(os.path.splitext(f)[0] + '.csv')
        c.render_mask(os.path.splitext(f)[0] + '_mask.tif')
        print('')
        return c


@graceful_exit('\x1b[1;31;40m' + 'ERROR: Could not load segmentation backend.' + '\x1b[0m')
def _get_backend(unet: bool, cellpose: bool, device: str):
    """ Get the appropriate Backend """
    if unet and not cellpose:
        backend = UNetWatershed(unet_model_path='/media/DataStorage/Dropbox (Partners HealthCare)/'
                                                'HairCellInstance/checkpoints/'
                                                'May11_16-38-12_chris-MS-7C37_checkpoint.hcnet',
                                frcnn_model_path='/media/DataStorage/Dropbox (Partners HealthCare)/HcUnet/TrainedModels/'
                                                 'fasterrcnn_Oct14_06:05.pth',
                                device=device).requires_grad_(False)
    elif not unet and not cellpose:
        backend = SpatialEmbedding(model_loc='/media/DataStorage/Dropbox (Partners HealthCare)/HairCellInstance/'
                                             'models_lts/spatial_embedding.trch',
                                   sigma=torch.tensor([0.2, 0.2, 0.2]),
                                   scale=25,
                                   device=device).requires_grad_(False)
    elif cellpose and not unet:
        from hcat.backends.cellpose import Cellpose  # cellpose announces it makes a log file
        backend = Cellpose(device=device)

    return backend
