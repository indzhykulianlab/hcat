import torch
from typing import List, Tuple, Optional, Union
from explore_lif import Reader
from transforms import _crop
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt


class ShapeError(Exception):
    pass


def calculate_indexes(pad_size: int, eval_image_size: int,
                      image_shape: int, padded_image_shape: int) -> List[List[int]]:
    """
    This calculates indexes for the complete evaluation of an arbitrarily large image by unet.
    each index is offset by eval_image_size, but has a width of eval_image_size + pad_size * 2.
    Unet needs padding on each side of the evaluation to ensure only full convolutions are used
    in generation of the final mask. If the algorithm cannot evenly create indexes for
    padded_image_shape, an additional index is added at the end of equal size.

    :param pad_size: int corresponding to the amount of padding on each side of the
                     padded image
    :param eval_image_size: int corresponding to the shape of the image to be used for
                            the final mask
    :param image_shape: int Shape of image before padding is applied
    :param padded_image_shape: int Shape of image after padding is applied

    :return: List of lists corresponding to the indexes
    """

    # We want to account for when the eval image size is super big, just return index for the whole image.
    if eval_image_size + (2 * pad_size) > image_shape:
        return [[0, image_shape - 1]]

    try:
        ind_list = torch.arange(0, image_shape, eval_image_size)
    except RuntimeError:
        raise RuntimeError(f'Calculate_indexes has incorrect values {pad_size} | {image_shape} | {eval_image_size}:\n'
                           f'You are likely trying to have a chunk smaller than the set evaluation image size. '
                           'Please decrease number of chunks.')
    ind = []
    for i, z in enumerate(ind_list):
        if i == 0:
            continue
        z1 = int(ind_list[i - 1])
        z2 = int(z - 1) + (2 * pad_size)
        if z2 < padded_image_shape:
            ind.append([z1, z2])
        else:
            break
    if not ind:  # Sometimes z is so small the first part doesnt work. Check if z_ind is empty, if it is do this!!!
        z1 = 0
        z2 = eval_image_size + pad_size * 2
        ind.append([z1, z2])
        ind.append([padded_image_shape - (eval_image_size + pad_size * 2), padded_image_shape])
    else:  # we always add at the end to ensure that the whole thing is covered.
        z1 = padded_image_shape - (eval_image_size + pad_size * 2)
        z2 = padded_image_shape - 1
        ind.append([z1, z2])
    return ind


@torch.jit.script
def crop_to_identical_size(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Crops Tensor a to the shape of Tensor b, then crops Tensor b to the shape of Tensor a.

    :param a: torch.
    :param b:
    :return:
    """
    if a.ndim != b.ndim:
        raise RuntimeError('Number of dimensions of tensor "a" does not equal tensor "b".')

    if a.ndim > 5:
        raise RuntimeError('Only supports tensors with minimum 3dimmensions and shape [..., X, Y, Z]')

    a = _crop(a, x=0, y=0, z=0, w=b.shape[-3], h=b.shape[-2], d=b.shape[-1])
    b = _crop(b, x=0, y=0, z=0, w=a.shape[-3], h=a.shape[-2], d=a.shape[-1])
    return a, b

# ########################################################################################################################
# #                                                       Postprocessing
# ########################################################################################################################

@torch.jit.script
def remove_edge_cells(mask: torch.Tensor) -> torch.Tensor:
    """
    Removes cells touching the border

    :param mask: (B, X, Y, Z)
    :return: mask (B, X, Y, Z)
    """

    if mask.ndim != 4: raise RuntimeError('input.ndim != 4')

    # Mask is empty, nothing to do.
    if mask.max() == 0:
        return mask

    left = torch.unique(mask[:, 0, :, :])
    right = torch.unique(mask[:, -1, :, :])
    top = torch.unique(mask[:, :, 0, :])
    bottom = torch.unique(mask[:, :, -1, :])

    cells = torch.unique(torch.cat((left, right, top, bottom)))

    for c in cells:
        if c == 0:
            continue
        mask[mask == c] = 0

    return mask


@torch.jit.script
def remove_wrong_sized_cells(mask: torch.Tensor) -> torch.Tensor:
    """
    Removes cells with outlandish volumes. These have to be wrong.

    :param mask: [B, C=1, X, Y, Z] int torch.Tensor: cell segmentation mask where each cell has a unique cell id
    :return:
    """
    unique = torch.unique(mask)
    unique = unique[unique.nonzero()]

    for u in unique:
        if (mask == u).sum() < 4000:
            mask[mask == u] = 0
        elif (mask == u).sum() > 30000:
            mask[mask == u] = 0
    return mask


# ########################################################################################################################
# #                                                       U Net Specific
# ########################################################################################################################

def pad_image_with_reflections(image: torch.Tensor, pad_size: Tuple[int] = (30, 30, 6)) -> torch.Tensor:
    """
    Pads image according to Unet spec
    expect [B, C, X, Y, Z]
    Adds pad size to each side of each dim. For example, if pad size is 10, then 10 px will be added on top, and on bottom.

    :param image:
    :param pad_size:
    :return:
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f'Expected image to be of type torch.tensor not {type(image)}')
    for pad in pad_size:
        if pad % 2 != 0:
            raise ValueError('Padding must be divisible by 2')

    device = image.device

    image_size = image.shape
    pad_size = np.array(pad_size)

    left_pad = image.cpu().numpy()[:, :, pad_size[0] - 1::-1, :, :]
    left_pad = torch.as_tensor(left_pad.copy())
    right_pad = image.cpu().numpy()[:, :, -1:-pad_size[0] - 1:-1, :, :]
    right_pad = torch.as_tensor(right_pad.copy())
    image = torch.cat((left_pad, image.cpu(), right_pad), dim=2).to(device)

    left_pad = 0
    right_pad = 0

    bottom_pad = image.cpu().numpy()[:, :, :, pad_size[1] - 1::-1, :]
    bottom_pad = torch.as_tensor(bottom_pad.copy())
    top_pad = image.cpu().numpy()[:, :, :, -1:-pad_size[1] - 1:-1, :]
    top_pad = torch.as_tensor(top_pad.copy())
    image = torch.cat((bottom_pad, image.cpu(), top_pad), dim=3).to(device)
    bottom_pad = 0
    top_pad = 0

    bottom_pad = image.cpu().numpy()[:, :, :, :, pad_size[2] - 1::-1]
    bottom_pad = torch.as_tensor(bottom_pad.copy())
    top_pad = image.cpu().numpy()[:, :, :, :, -1:-pad_size[2] - 1:-1]
    top_pad = torch.as_tensor(top_pad.copy())

    return torch.cat((bottom_pad, image.cpu(), top_pad), dim=4).to(device)


########################################################################################################################
#                                                         Generics
########################################################################################################################


def load(file: str, header_name: Optional[str] = 'TileScan 1 Merged',
         verbose: bool = False) -> Union[None, np.array]:
    """
    Loads image file (*leica or *tif) and returns an np.array

    :param file: str path to the file
    :param header_name: Optional[str] header name of lif. Does nothing image file is a tif
    :param verbose: bool print status of image loading to standard out.
    :return: np.array image matrix from file, aborts if the image is too large and returns None.
    """

    image_base = None

    if file.endswith('.lif'):  # Load lif file
        reader = Reader(file)
        series = reader.getSeries()
        for i, header in enumerate(reader.getSeriesHeaders()):
            if header.getName() == header_name:  ###'TileScan 1 Merged':

                if verbose:
                    print(f'Loading {file}...')

                chosen = series[i]
                for c in range(4):
                    if c == 0:
                        image_base = chosen.getXYZ(T=0, channel=c)[np.newaxis]

                        if verbose:
                            print(f'Estimated Image Numel: {image_base.size * 4}')

                        if image_base.size * 4 > 9000 * 9000 * 40 * 4:
                            return None

                    else:
                        image_base = np.concatenate((image_base, chosen.getXYZ(T=0, channel=c)[np.newaxis]), axis=0)
                del series, header, chosen

    elif file.endswith('.tif'): # Load a tif
        if verbose:
            print(f'Loading {file}...')

        image_base = io.imread(file)

        if image_base.ndim == 4:
            image_base = image_base.transpose((-1, 1, 2, 0))

        elif image_base.ndim == 3:  # Suppose you load a 3D image with one channel.
            image_base = image_base[np.newaxis, ...]
            image_base = np.concatenate((image_base, image_base, image_base, image_base), axis=0).transpose(0, 2, 3, 1)
        else:
            raise RuntimeError(f'Image ndim not 3 or 4, {image_base.ndim}, {image_base.shape}')
    else:
        raise NotImplementedError(f'Cannot Load file of with this extension: {file}')

    return image_base


########################################################################################################################
#                                                         Plotting
########################################################################################################################

def plot_embedding(embedding: torch.Tensor, centroids: torch.Tensor) -> None:
    num = 25
    x = embedding.detach().cpu().numpy()[0, 0, ...].flatten() * num
    y = embedding.detach().cpu().numpy()[0, 1, ...].flatten() * num
    plt.hist2d(y, x, bins=(embedding.shape[2], embedding.shape[3]))
    print(centroids)
    plt.plot(centroids[0, :, 1].cpu().numpy(), centroids[0, :, 0].cpu().numpy(), 'ro')

    plt.show()
