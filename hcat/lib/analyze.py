from typing import *

import torch
import torchvision.transforms.functional
from torch import Tensor

from hcat.lib.animal import *
from hcat.lib.frequency import (
    interpolate_path,
    tonotopy_to_percentage,
    coord_to_percentage,
    calculate_frequency,
)
from hcat.state import *

animal_map = {"mouse": Mouse}  # from hcat.lib.animal


def assert_validity(cochlea: Cochlea) -> Tuple[bool, str | None]:
    # must have at least one piece
    # each piece must have same animal
    # each piece must have an eval region
    # each piece may not necessarily have the same pixel sizes...., Thats ok...
    # each piece must have a freq region set...

    if len(cochlea.children) < 1:
        return False, "ERROR: Cochlea must have at least one piece."

    selected_piece: Piece = cochlea.get_selected_child()
    animal = selected_piece.animal
    if not all(
        [p.animal == animal for p in cochlea.children.values() if isinstance(p, Piece)]
    ):
        return False, "ERROR: All pieces must be of the same animal."

    # if not all(
    #     [p.eval_region for p in cochlea.children.values() if isinstance(p, Piece)]
    # ):
    #     return False, "ERROR: All pieces must have a user defined evaluation region."

    if not all(
        [
            p.freq_path is not None
            for p in cochlea.children.values()
            if isinstance(p, Piece)
        ]
    ):
        return False, "ERROR: All pieces must have a user defined tonotopic path."

    # if we've gotten this far, its all good
    return True, "VALID"

# args kwargs need to be there to potentially make cython happy
def assign_frequency(state_item: Cochlea | Piece | Cell) -> Cochlea:
    """
    Given any state_item we can assign
    Assumes that the cochlea has ALL cells already predicted...
    and that all pieces have a freq region.

    Makes changes in place, but returns out of convenience...

    :param cochlea: current gui state after all cell predictions have been assigned
    :return: cochlea.
    """
    if not isinstance(state_item, Cochlea):
        cochlea = state_item.parent
        while not isinstance(cochlea, Cochlea):
            cochlea = cochlea.parent
            if cochlea is None:
                raise RuntimeError('state_item parent should never be none...')
    else:
        cochlea = state_item

    print('inside worker: ', state_item)

    pieces: List[Piece] = [p for p in cochlea.children.values() if isinstance(p, Piece)]
    pieces.sort(key=lambda p: p.relative_order)  # sort by order

    paths: List[List[Tuple[float, float]]] = [
        interpolate_path(p.freq_path) for p in pieces
    ]

    percentages, total_length = tonotopy_to_percentage(
        paths, px_sizes=[p.pixel_size_xy for p in pieces]
    )

    assert (
        len(pieces) == len(paths) == len(percentages)
    ), "different number of pieces, paths, and percentages..."

    for piece, path, percentage in zip(pieces, paths, percentages):

        if isinstance(state_item, Piece):
            if not state_item is piece:
                continue

        cells = [c for c in piece.children.values() if isinstance(c, Cell)]
        cells.extend([c for c in piece._candidate_children.values() if isinstance(c, Cell)])

        for c in cells:
            if isinstance(state_item, Cell):
                if not state_item is c:
                    continue


            cell_percent = coord_to_percentage(
                coordinate=c.get_cell_center(), path=path, percentage=percentage
            )
            c.set_percentage(cell_percent)
            c.set_distance(cell_percent * total_length)
            freq = calculate_frequency(cell_percent, animal_map[piece.animal.lower()])
            c.set_frequency(freq)

        piece.set_basal_freq(calculate_frequency(percentage[0], animal_map[piece.animal.lower()]))
        piece.set_apical_freq(calculate_frequency(percentage[-1], animal_map[piece.animal.lower()]))

    return cochlea


def rescale_to_optimal_size(
    image: torch.Tensor,
    px_size: float,
    antialias: Optional[bool] = True,
) -> Tensor:
    """
    Upscales or downscales an torch.Tensor image to the optimal size for HCAT detection.
    Scales to 288.88nm/px based on cell_diameter, or current_pixel_size. If neither the current pixel size, or
    cell diameter were passed, this function returns the original image.

    Shapes:
        - image: :math:`( X_{in}, Z_{in}, C = 3)`

    :param image: Image to be resized
    :param cell_diameter: Diameter of cells in unscaled image
    :param current_pixel_size: Pixel size of unscaled image
    :param antialias: If true, performs antialiasing upon scaling
    :param verbose: Prints operation to standard out

    :return: Scaled image
    """

    OPTIMAL = 289.0
    if abs(OPTIMAL - px_size) < 2:  # close enough to be a no-op
        return image

    scale = px_size / OPTIMAL
    x, y, c = image.shape
    new_size = [round(x * scale), round(y * scale)]

    image = torchvision.transforms.functional.resize(
        image.permute(2, 0, 1), new_size, antialias=antialias
    )

    return image.permute(1, 2, 0)
