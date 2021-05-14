import time

from src.utils import calculate_indexes, remove_edge_cells, remove_wrong_sized_cells
import src.functional
from src.cell import Cell
from src.cochlea import Cochlea

from src.backends.spatial_embedding import SpatialEmbedding
from src.backends.unet_and_watershed import UNetWatershed

import torch
import click
from tqdm import tqdm
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
from src.explore_lif import Reader, get_xml

import glob
import os.path
import gc
import psutil
from typing import Optional, List

# default_path = '/media/chris/Padlock_3/Dose-adjusted injection data/**/**/'
default_path = '/media/DataStorage/Dropbox (Partners HealthCare)/HairCellInstance/data/test/shit'
# default_path = '/media/DataStorage/Dropbox (Partners HealthCare)/HairCellInstance/data/'
network = '/run/user/1000/gvfs/smb-share:server=apollo,share=research/ENT/Indzhykulian/David Rosenberg/AAV injection imaging/'


@click.command()
@click.argument('f', default=None)
@click.option('--render', is_flag=True)
@click.option('--channel', default=2, help='Channel index to segment')
@click.option('--dtype', default=None, help='dtype of input image')
@click.option('--unet', is_flag=True)
@click.option('--figure', is_flag=True)
def analyze(f: str, channel: int, render: bool,
            dtype: Optional[str], unet: bool,
            figure: Optional[bool]) -> None:
    if f is None:
        print('\x1b[1;31;40m' + 'ERROR: No File to Analyze... \nAborting.' + '\x1b[0m')
        return None

    with torch.no_grad():
        if unet:
            backend = UNetWatershed(unet_model_path='/media/DataStorage/Dropbox (Partners HealthCare)/'
                                                 'HairCellInstance/checkpoints/'
                                                 'May11_16-38-12_chris-MS-7C37_checkpoint.hcnet',
                                    frcnn_model_path='/media/DataStorage/Dropbox (Partners HealthCare)/HcUnet/TrainedModels/'
                                                     'fasterrcnn_Oct14_06:05.pth').requires_grad_(False)

        elif not unet:
            backend = SpatialEmbedding(model_loc='/media/DataStorage/Dropbox (Partners HealthCare)/'
                                                     'HairCellInstance/checkpoints/'
                                                     'May04_18-31-35_chris-MS-7C37_checkpoint.hcnet',
                                       sigma=torch.tensor([0.2, 0.2, 0.2]),
                                       scale=25,
                                       figure=figure).requires_grad_(False)

        image_base = src.utils.load(f, 'TileScan 1 Merged', verbose=True)

        if image_base is None:  # Break if couldn't load stuff!
            return None

        print(
            f'DONE: shape: {image_base.shape}, min: {image_base.min()}, max: {image_base.max()}, dtype: {image_base.dtype}')


        dtype = image_base.dtype if dtype is None else dtype
        if dtype == 'uint16':
            scale = 2**16
        elif dtype == 'uint8':
            scale = 2 ** 8
        elif dtype == 'uint12': #FM143 usually is this...
            scale = 2 ** 12
        else:
            raise ValueError(f'No scale factor for unrecognized dtype: {dtype}')

        # c needs to be [Blue, Green, Yellow, Red]
        c, x, y, z = image_base.shape

        out_img = torch.zeros((1, x, y, z), dtype=torch.long)  # (1, X, Y, Z)

        max_cell: int = 0

        x_ind: List[List[int]] = calculate_indexes(30, 255, x, x)  # [[0, 255], [30, 285], ...]
        y_ind: List[List[int]] = calculate_indexes(30, 255, y, y)  # [[0, 255], [30, 285], ...]
        total: int = len(x_ind) * len(y_ind)

        cells = []
        print('Segmenting...')
        for (x, y) in tqdm(product(x_ind, y_ind), total=total):

            # Load and prepare image for ML model evaluation
            image = image_base[:, x[0]:x[1], y[0]:y[1]].astype(np.int32)
            image = torch.from_numpy(image).float().div(scale).unsqueeze(0)
            image = image.div(0.5) if image.min() < 0 else image.sub(0.5).div(0.5)  # may be unsigned or signed int

            out = backend(image.cuda()) if unet else backend(image[:, [channel], ...].cuda())

            if out.shape[1] == 0:
                continue

            # postprocess
            value, out = out.max(1)
            out[value < 0.25] = 0

            out = remove_wrong_sized_cells(out)
            out = remove_edge_cells(out).cpu()

            out[out > 0] = out[out > 0] + max_cell
            max_cell = out.max() if out.max() > 0 else max_cell

            # merging!
            x_exp = torch.tensor([x[0], x[1]])
            y_exp = torch.tensor([y[0], y[1]])
            x_exp[x_exp < 0] = 0
            y_exp[y_exp < 0] = 0
            destination = out_img[:, x_exp[0]:x_exp[1], y_exp[0]:y_exp[1], :]
            destination, new_ids = src.functional.merge_regions(destination.to(out.device), out, 0.35)

            out_img[:, x_exp[0]:x_exp[1], y_exp[0]:y_exp[1], :] = destination.cpu()

            for u in new_ids:
                if u == 0:
                    continue
                center = (out == u).nonzero(as_tuple=False).float().mean(0)
                center[1] += x[0]
                center[2] += y[0]
                cells.append(Cell(id=u,
                                  image=torch.clone(image[:, :, :-1:, :-1:, 0:out.shape[-1]].mul(0.5).add(0.5)),
                                  mask=(out == u).unsqueeze(0),
                                  loc=center))
        print('Predicted Number of Cells: ', len(cells))

        mem = psutil.virtual_memory()  # get memory for later

        del image_base  # memory issues are tight. removing image and image_base help
        gc.collect()

        equal_spaced_points, percentage, apex = src.functional.get_cochlear_length(out_img, equal_spaced_distance=0.01)

        # Figure out the frequency location of each cell
        for c in cells:
            c.calculate_frequency(equal_spaced_points)

        xml = get_xml(f) if f.endswith('.lif') else None

        # Store
        c = Cochlea(mask=out_img,
                    filename=f,
                    leica_metadata=xml,
                    im_shape=out_img.shape,
                    cochlear_percent=percentage,
                    curvature=equal_spaced_points,
                    cells=cells,
                    apex=apex)


        # This takes too much memory for huge cochleas
        THRESHOLD = 20 * 1024 * 1024 * 1024  # 15GB
        if mem.available > THRESHOLD:
            c.compress_mask(vebose=True)
        else:
            print('\x1b[0;31;40m' + 'Warning: Low free system memory. Cannot compress output... ' + '\x1b[0m')

        c.save(os.path.splitext(f)[0] + '.cochlea')
        print('Saved: ', os.path.splitext(f)[0] + '.cochlea')

        try:
            c.make_fig(os.path.splitext(f)[0] + '.pdf')
        except:
            print('\x1b[1;31;40m' + 'ERROR: Figure Render failed.' + '\x1b[0m')

        try:
            c.write_csv(os.path.splitext(f)[0] + '.csv')
        except:
            print('\x1b[1;31;40m' + 'ERROR: csv generation failed.' + '\x1b[0m')

        if render:
            c.render_mask(os.path.splitext(f)[0] + '_mask.tif')

if __name__ == '__main__':
    analyze()
