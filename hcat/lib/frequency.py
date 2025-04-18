import math
from itertools import accumulate
from typing import Tuple, Optional, List

import numpy as np
import torch
from torch import Tensor

from hcat.lib.animal import Animal
from hcat.state import Cell


def interpolate_path(
    path: List[Tuple[float, float]],
    equal_spaced_distance: Optional[float] = 1,
    as_array: bool = False,
) -> List[Tuple[float, float]] | np.ndarray:
    """
    Given a sparse path, will interpolate the path to equal spaced points...

    :param path:  List of points on path [[x0, y0], [x1, y1], ..., [xn, yn]] for n points
    :param equal_spaced_distance:  interpoation factor
    :param as_array: if true returns as a numpy array

    :return: List of lines connecting the point and the path [[x0, y0, x1, y1], ...] if as_array is false, otherwise a
    numpy array
    """
    path = np.array(path)  # N, 2

    curvature = []
    for l0, l1 in zip(path[:-1:], path[1::]):
        # l0, l1 = [x, y], [x, y]

        w = l1[0] - l0[0]
        h = l1[1] - l0[1]
        # slope = h / (w + 1e-10)
        distance: float = math.sqrt(w**2 + h**2)
        N_segments: int = round(distance / equal_spaced_distance)

        dx: float = w / N_segments
        dy: float = h / N_segments

        x: List[float] = [l0[0] + i * dx for i in range(N_segments)]
        y: List[float] = [l0[1] + i * dy for i in range(N_segments)]
        curvature.append(np.array([x, y]).T)  # T because this constructor does 2, N

    curvature = np.concatenate(curvature, axis=0)
    if not as_array:
        curvature = curvature.tolist()

    return curvature


def closest_point(
    cells: List[Cell], path: np.array
) -> List[Tuple[float, float, float, float]]:
    """
    Given a path of points, as defined by a np.array, for each cell's center,
    calculates the closest point in that path.

    :param cells: List[Cells] with atribute bbox, cell.bbox = [x0, y0, x1, y1]
    :param path: np.array with shape [N, C=2] of N points, where C => [xn, yn]
    :return: List of lines, in the same order as cells, of each closest point. [x_start,y_start,x_end,y_end]
    """
    # assert len(cells) > 0, f'{len(cells)=} should not be zero'
    bboxes = np.array([c.bbox for c in cells])  # N, 4 ,[x0, y0, x1, y1]
    centers = np.zeros((bboxes.shape[0], 2))

    centers[:, 0] = (bboxes[:, 2] - bboxes[:, 0]) / 2 + bboxes[:, 0]  # cx
    centers[:, 1] = (bboxes[:, 3] - bboxes[:, 1]) / 2 + bboxes[:, 1]  # cy

    centers: torch.tensor = torch.from_numpy(centers)  # N 2
    path: torch.tensor = torch.from_numpy(np.array(path))  # M 2

    dist = torch.cdist(centers, path)  # N, M
    argmin = dist.argmin(1)
    # print(f'{path.shape=}, {dist.shape=}, {argmin.shape=}')

    closest_path = path[argmin, :]
    lines = torch.concatenate((centers, closest_path), dim=1).numpy().tolist()

    return lines


def tonotopy_to_percentage(
    curves: List[List[Tuple[float, float]]], px_sizes: List[float]
) -> Tuple[List[List[float]], float]:
    """
    Calculates a total percentage of the distance the cochleae represent

    :param curves: List of tonotopy curves [x,y]
    :param px_sizes: List of pixel sizes in nm

    :return: List of percentages...
    """
    if len(curves) != len(px_sizes):
        raise RuntimeError("len(curves) != len(px_sizes)")

    distance = lambda p0, p1: math.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)

    distances = []  # start at zero...
    for curve, px in zip(curves, px_sizes):
        if len(curve) <= 0: raise RuntimeError('curve len is 0')
        curve = [(x * px, y * px) for (x, y) in curve]
        distances.append(
            list(accumulate(distance(curve[i - 1], curve[i]) for i in range(1, len(curve))))
        )

    # add get total distances...
    for i, dist in enumerate(distances):
        if i <= 0:
            distances[i] = [0] + dist
        else:
            last = distances[i-1][-1]
            dist = [d + last for d in dist]
            distances[i] = [distances[i-1][-1]] + dist


    # accumulate is a generator...

    total_length = -1
    for _dist in distances:
        _max = max(_dist)
        total_length = _max if _max > total_length else total_length

    percentages: List[List[float]] = []
    for _dist in distances:

        _per = [t / total_length for t in _dist]
        _per = [0.0] + _per
        percentages.append(_per)

    return percentages, total_length


def coord_to_percentage(
    coordinate: Tuple[float, float],
    path: List[Tuple[float, float]],
    percentage: List[float],
) -> float:
    """
    Given a coordinate, representing the center of a predicted bounding box, and a user defined path
    along a cochlea, calculate where that center is along that path.

    :param coordinate: [center_x, center_y]
    :param path: [(x0, y0), (x1, y1), ..., (xn, yn)]
    :param percentage: percentages 0->1 corresponding to each coordinate in path [p0, p1, ..., pn]

    :return: percentage long length of path that coordinate lies
    """
    if len(coordinate) > 2:
        raise RuntimeError("Coordinate must be a tuple: (x, y)")

    coordinate: Tensor = torch.tensor(coordinate).unsqueeze(0)
    path: Tensor = torch.tensor(path)  # M 2

    dist: Tensor = torch.cdist(coordinate, path)  # N, M
    argmin = dist.argmin(1)
    per = percentage[argmin.item()]
    # print(f'{path.shape=}, {dist.shape=}, {argmin.shape=}')
    return per


def calculate_frequency(distance_percentage: float, animal: Animal) -> float:
    """
    Returns the frequency at a certain percentage along the tonotopic axis for an animal.

    :param distance_percentage: percentage of length of cochlea 0->1
    :param animal: animal enum from hcat.lib.animal
    :return: frequency in Hz
    """
    if animal.FN is None:
        return animal.A * (animal.BASE ** ((1 - distance_percentage) * animal.B) - animal.K)
    else:
        return animal.FN(distance_percentage)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    a = [[0, 1], [1, 2], [1, 3], [0, 4]]
    _a = np.array(a)
    plt.plot(_a[:, 0], _a[:, 1])

    out = interpolate_path(a, 0.001, as_array=True)

    plt.plot(out[:, 0], out[:, 1])

    plt.show()

    cells = [Cell(id=0, bbox=[1.5, 1.5, 9.5, 3.5], parent=None, score=1.0)]

    test = closest_point(cells, out)
    print(test)
