import os.path
from math import sqrt
from typing import List, Tuple, Optional
from xml.etree import ElementTree as ET

import torch
from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor


def calculate_1D_indexes(image_shape: int,
                         eval_image_size: int,
                         padding: int) -> List[List[int]]:
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
    padded_image_shape = image_shape + 2 * padding

    # We want to account for when the eval image size is super big, just return index for the whole image.
    if eval_image_size + (2 * padding) > image_shape:
        return [[0, image_shape - 1]]

    try:
        ind_list = torch.arange(0, image_shape, eval_image_size)
    except RuntimeError:
        raise RuntimeError(f'Calculate_indexes has incorrect values {padding} | {image_shape} | {eval_image_size}:\n'
                           f'You are likely trying to have a chunk smaller than the set evaluation image size. '
                           'Please decrease number of chunks.')
    ind = []
    for i, z in enumerate(ind_list):
        if i == 0:
            continue
        z1 = int(ind_list[i - 1])
        z2 = int(z - 1) + (2 * padding)
        if z2 < padded_image_shape:
            ind.append([z1, z2])
        else:
            break
    if not ind:  # Sometimes z is so small the first part doesnt work. Check if z_ind is empty, if it is do this!!!
        z1 = 0
        z2 = eval_image_size + padding * 2
        ind.append([z1, z2])
        ind.append([padded_image_shape - (eval_image_size + padding * 2), padded_image_shape])
    else:  # we always add at the end to ensure that the whole thing is covered.
        z1 = padded_image_shape - (eval_image_size + padding * 2)
        z2 = padded_image_shape - 1
        ind.append([z1, z2])
    return ind

def calculate_2D_indicies(image_shape: List[int],
                          eval_iamge_size: Optional[List[int]] = [256, 256],
                          padding: Optional[List[int]] = [0, 0]) -> Tuple[List[List[int]]]:

    _, x, y = image_shape
    x_eval = eval_iamge_size[0]
    y_eval = eval_iamge_size[1]

    x_pad, y_pad = padding

    x_ind = calculate_1D_indexes(x, x_eval, x_pad)
    y_ind = calculate_1D_indexes(y, y_eval, y_pad)

    return x_ind, y_ind

def write_predictions(boxes: torch.Tensor,
                      labels: torch.Tensor,
                      filename: str):

    encoding = {
        1: 'OHC',
        2: 'IHC'
    }

    folder = os.path.split(filename)[0]
    folder = os.path.split(folder)[-1] if len(folder) != 0 else os.path.split(os.getcwd())[1]

    root = ET.Element('annotation')
    ET.SubElement(root, 'folder').text = folder
    ET.SubElement(root, 'filename').text = filename
    ET.SubElement(root, 'path').text = 'unspecified'
    source = ET.SubElement(root, 'source')
    ET.SubElement(source, 'database').text = 'unknown'
    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = 'unspecified'
    ET.SubElement(size, 'height').text = 'unspecified'
    ET.SubElement(size, 'depth').text = str(1)
    ET.SubElement(root, 'segmented').text = '0'

    for box, label in zip(boxes, labels):
        x0, y0, x1, y1 = box
        #  xml write xmin, xmax, ymin, ymax
        object = ET.SubElement(root, 'object')
        ET.SubElement(object, 'name').text = encoding[label.item()]
        ET.SubElement(object, 'pose').text = 'Unspecified'
        ET.SubElement(object, 'truncated').text = '0'
        ET.SubElement(object, 'difficult').text = '0'
        bndbox = ET.SubElement(object, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(int(x0))
        ET.SubElement(bndbox, 'ymin').text = str(int(y0))
        ET.SubElement(bndbox, 'xmax').text = str(int(x1))
        ET.SubElement(bndbox, 'ymax').text = str(int(y1))

    tree = ET.ElementTree(root)
    tree.write(filename)



def qpointf_dist(a: QPointF, b: QPointF) -> float:
    return sqrt((a.x() - b.x()) ** 2 + (a.y() - b.y()) ** 2)



def hex_to_qcolor(hex_color):
    """Converts a string hex color code to a QColor object.

    """
    # Remove the "#" symbol if it exists
    hex_color = hex_color.lstrip("#")

    # Convert the hex color to its RGB components
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    # Create and return a QColor object
    return QColor(r, g, b)

def hex_to_qcolor(hex_color) -> QColor:
    """Converts a string hex color code to a QColor object.

    """
    # Remove the "#" symbol if it exists
    hex_color = hex_color.lstrip("#")

    # Convert the hex color to its RGB components
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    # Create and return a QColor object
    return QColor(r, g, b)


def qcolor_to_hex(qcolor: QColor) -> str:
    """Converts a QColor object to a string hex color code.

    Args:
        qcolor (QColor): A QColor object representing a color.

    Returns:
        A string representing the equivalent hex color code (e.g. "#FF0000").
    """
    # Get the RGB components of the QColor object
    r, g, b, a = qcolor.getRgb()

    # Convert the RGB components to their hex equivalents
    hex_r = hex(r)[2:].zfill(2)
    hex_g = hex(g)[2:].zfill(2)
    hex_b = hex(b)[2:].zfill(2)

    # Create and return the hex color code string
    return "#" + hex_r + hex_g + hex_b


if __name__ == '__main__':
    x_ind, y_ind = calculate_2D_indicies([3, 1000, 1000], [256, 256], [0, 0])
    for x in x_ind:
        for y in y_ind:
            print(x, y)