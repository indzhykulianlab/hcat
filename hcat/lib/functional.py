import pandas
from sklearn.cluster import DBSCAN
import torch
from torch import Tensor
import hcat.lib.utils
from hcat import ShapeError

from typing import Tuple, Dict, Optional, List

import numpy as np
import skimage
import skimage.exposure
import skimage.filters
import skimage.morphology
import skimage.feature
import skimage.segmentation
import skimage.transform
import skimage.feature

from hcat.lib.utils import _crop
from hcat.lib.cell import Cell
from hcat.lib.utils import graceful_exit

import scipy.ndimage
import scipy.ndimage.morphology
from scipy.interpolate import splprep, splev
import GPy

import torchvision.ops
import torch.nn as nn

import matplotlib.pyplot as plt


@torch.jit.script
def _iou(a: Tensor, b: Tensor) -> Tensor:
    """
    Takes two identically sized tensors and computes IOU of the Masks
    Assume a, b are of the same shape and boolean tensors with values 0 and 1

    :param a: Tensor - Bool
    :param b: Tensor - Bool

    :return: IoU
    """
    if a.dtype != torch.float: raise ValueError('input tensors must be of type float')
    val = (a + b)
    intersection = val.gt(1).sum()
    union = val.gt(0).sum()
    return intersection / union


class nms(nn.Module):
    def __init__(self):
        super(nms, self).__init__()

    def forward(self, mask: Tensor, threshold: Optional[float] = 0.5) -> Tensor:
        """
        Performs non-maximum supression on object probability masks.

        Example:

        >>> from hcat.lib.functional import nms
        >>> import torch
        >>> cell_mask = torch.load('cell_segmentation_masks.trch') #  [B, N, X, Y, Z]
        >>> nms = torch.jit.script(nms()) # [1, 20, 100, 100, 10]
        >>> keep = nms(cell_mask)
        >>> cell_mask = cell_mask[:, keep, ...] # [1, 15, 100, 100, 10]

        :param mask: [B, N, X, Y, Z] predicted cell instance segmentation mask
        :param threshold: iou rejection threshold
        :return: [N] index of remaining cells
        """
        n = mask.shape[1]
        if n == 0:
            return torch.zeros(n).gt(0)

        if mask.dtype != torch.float:
            raise ValueError('Mask dtype must be float', mask.dtype)

        max_instances = mask.shape[1]
        ind = torch.ones(max_instances, dtype=torch.int, device=mask.device)
        score = mask.squeeze(0).reshape(max_instances, -1).sum(-1)

        for i in range(max_instances):
            if ind[i] == 0:
                continue
            for j in range(max_instances):
                iou = _iou(mask[:, i, ...].float(), mask[:, j, ...].float())
                if iou > threshold:
                    if score[j] > score[i]:
                        ind[i] = 0
                    if score[i] > score[j]:
                        ind[j] = 0
                        break  # break out of loop because base is now zero!

        return ind > 0


class PredictCurvature:
    def __init__(self,
                 voxel_dim: Optional[Tuple[float, float, float]] =(.28888, .28888, .2888 * 3), # um
                 equal_spaced_distance: Optional[float] = 0.01,
                 erode: Optional[int] = 3,
                 scale_factor: Optional[int] = 10,
                 method: Optional[str] = None):
        """
        Initialize the cochlear path prediction algorithm.

        This module will attempt to fit a set of equally spaced points to an image containing a whole cochlea in one contiguous piece.
        Will attempt to calculate via myo7a signal, but may also use box detections.

        :param voxel_dim: Tuple of pixel spacings in um
        :param equal_spaced_distance: How far apart individual points of the resulting line need to be. Safely set at 0.1.
        :param erode: How many times a binary erosion should be performed on the MyoVIIA signal. Larger values can reduce false positive values.
        :param scale_factor: Downscale factor for curve estimation. Smaller values incur a performance hit.
        :param method: Cochlea estimation approach: ['mask', 'maxproject', None]
        """

        self.equal_spaced_distance = equal_spaced_distance
        self.erode = erode
        self.method = method
        self.voxel_dim = voxel_dim
        self.scale_factor = scale_factor

        _methods = ['mask', 'maxproject', None]
        if not self.method in _methods:
            raise RuntimeError(f'Unknown method: {method}.')

    def __call__(self,
                 image: Optional[Tensor] = None,
                 cells: Optional[List[Cell]] = None,
                 csv: Optional[str] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Tries various methods to predict curvature.

        If user provides a csv of curvature default to this.
        In order -> User provided CSV, Myo7a, Mask

        :param image: torch.Tensor image of a complete cochlea in one contiguous piece.
        :param mask: A binary mask by which to fit a nonlinear curve through
        :param csv: A csv of manually chosen points by which to fit a spline through. Comperable to the "EPL Method" of frequency estimation.
        :return: Tuple[Tensor, Tensor, Tensor]:
            1) A tensor of equally spaced points which follows the cochlear path,
            2) A tensor representing the distance from the estimated apex of the cochlea
            3) The estimated apex of the cochlea
        """
        curvature, distance, apex = None, None, None

        if csv is not None:
            print('RUNNING EPL METHOD')
            curvature, distance, apex = self.fitEPL(csv)

        elif self.method is None:
            if csv is not None:
                curvature, distance, apex = self.fitEPL(csv)

            if curvature is None and cells is not None:
                try:
                    # Estimate cochlear curvature. Try the
                    _, x, y, _ = image.shape
                    mask = torch.zeros((1, x, y, 1))
                    for c in cells:
                        x0, y0, x1, y1 = torch.round(c.boxes).int()
                        mask[:, y0:y1, x0:x1, :] = 1
                    curvature, distance, apex = self.fit(mask, 'mask', diagnostic=False)

                except Exception:
                    curvature, distance, apex = None, None, None

            try_again = curvature is None and image is not None
            try_again = try_again or distance.max() < 4000

            if try_again:
                try:
                    curvature, distance, apex = self.fit(image, 'maxproject')
                except Exception:
                    curvature, distance, apex = None, None, None

        elif self.method == 'mask':
            _, x, y, _ = image.shape
            mask = torch.zeros((1, x, y, 1))
            for c in cells:
                x0, y0, x1, y1 = torch.round(c.boxes).int()
                mask[:, y0:y1, x0:x1, :] = 1
            curvature, distance, apex = self.fit(mask, 'mask', diagnostic=False)

        elif self.method == 'maxproject':
            curvature, distance, apex = self.fit(image, 'maxproject', diagnostic=False)

        else:
            raise ValueError(f'Unknown method: {self.method}')

        return curvature, distance, apex

    def fit(self, base: Tensor, method: str, diagnostic=False) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Predicts cochlear curvature from a predicted segmentation mask.

        Uses beta spline fits as well as a gaussian process regression to estimate a contiguous curve in spherical space.

        .. warning:
           Will not raise an error upon failure, instead returns None and prints to standard out

        :parameter image: [C, X, Y, Z] bool tensor
        :return: Tuple[Tensor, Tensor, Tensor]:
            - equal_spaced_points: [2, N] Tensor of pixel locations
            - percentage: [N] Tensor of cochlear percentage
            - apex: Tensor location of apex (guess)
        """

        if method == 'mask':
            image = base[0, ..., 0].bool().int()
            image = self._preprocess(image, diagnostic)
        elif method == 'stack':
            image = base[0, ...].sum(-1).gt(1.5)
            image = self._preprocess(image, diagnostic)
        elif method == 'maxproject':
            gt = 0.55
            max = 0
            while max == 0:
                image = base[0, ..., 0].gt(gt)
                image = self._preprocess(image, diagnostic)
                max = image.max()
                gt -= 0.15

        # Sometimes there are NaN or inf we have to take care of
        image[np.isnan(image)] = 0
        try:
            center_of_mass = np.array(scipy.ndimage.center_of_mass(image))
            while image[int(center_of_mass[0]), int(center_of_mass[1])] > 0:
                center_of_mass += 1
        except ValueError:
            center_of_mass = [image.shape[0], image.shape[1]]

        # Turn the binary image into a list of points for each pixel that isnt black
        x, y = image.nonzero()
        x += -int(center_of_mass[0])
        y += -int(center_of_mass[1])

        # Transform into spherical space
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(x, y)

        # sort by theta
        ind = theta.argsort()
        theta = theta[ind]
        r = r[ind]

        # there will be a break somewhere because the cochlea isnt a full circle
        # Find the break and subtract 2pi to make the fun continuous
        loc = np.abs(theta[0:-2:1] - theta[1:-1:1])

        # Correct if largest gap is larger than 20 degrees
        if np.abs(loc.max()) > 20 / 180 * np.pi:
            theta[loc.argmax()::] += -2 * np.pi
            ind = theta.argsort()[1:-1:1]
            theta = theta[ind]
            r = r[ind]

        # run a spline in spherical space after sorting to get a best approximated fit
        tck, u = splprep([theta, r], w=np.ones(len(r)) / len(r), s=0.5e-6, k=3)

        # Run a Gaussian Process Fit, seems to work better than spline
        kernel = GPy.kern.RBF(input_dim=1, variance=50., lengthscale=5.)  # 100 before

        # Might fail, throw an error
        if np.any(np.isnan(theta)) or np.any(np.isnan(r)) or np.any(np.isinf(theta)) or np.any(np.isinf(r)):
            print('\x1b[1;31;40m' + 'ERROR: Could not calculate cochlear length.' + '\x1b[0m')
            return None, None, None

        m = GPy.models.GPRegression(theta[::2, np.newaxis], r[::2, np.newaxis], kernel)
        m.optimize()

        theta = np.linspace(theta.min(), theta.max(), 10000)
        r_, _ = m.predict(theta[:, np.newaxis])
        r_ = r_[:, 0]
        theta_ = theta

        x_spline = r_ * np.cos(theta_) + center_of_mass[1]
        y_spline = r_ * np.sin(theta_) + center_of_mass[0]

        # x_spline and y_spline have tons and tons of data points.
        # We want equally spaced points corresponding to a certain distance along the cochlea
        # i.e. we want a point ever mm which is not guaranteed by x_spline and y_spline

        # THIS ONLY REMOVES STUFF, DOESNT INTERPOLATE!!!!!!
        curvature = []
        for i, coord in enumerate(zip(x_spline, y_spline)):
            if i == 0:
                base = coord
                curvature.append(base)
            if np.sqrt((base[0] - coord[0]) ** 2 + (base[1] - coord[1]) ** 2) > self.equal_spaced_distance:
                curvature.append(coord)
                base = coord

        curvature = np.array(curvature) * self.scale_factor # <-- Scale factor from above
        curvature = curvature.T

        # Figure out the knot with the larget value and which side its closest to. Thats probably the apex
        # If the curviest point is on the first half of the curve or second half. 1
        curve = tck[1][0]
        curviest_point = np.abs(curve).argmax()
        # curviest_point_a = np.abs(curve[:curve.shape[0]//3:]).argmax()
        # curviest_point_b = np.abs(curve[2*curve.shape[0]//3::]).argmax()
        #
        apex = curvature[:, -1] if curviest_point < curve.shape[0] // 2 else curvature[:, 0]
        # apex = curvature[:, -1] if curviest_point_b > curviest_point_a else curvature[:, 0]

        # sort curvature so that it ALWAYS GOES base to apex
        # curvature = curvature[:, ::-1] if curviest_point_b > curviest_point_a else curvature
        curvature = curvature[:, ::-1] if np.all(apex == curvature[:, -1]) else curvature

        # Calculate the distance from the apex
        distance = np.zeros(curvature.shape[1])
        x0, y0 = curvature[:, 0]
        for i in range(1, curvature.shape[1]):
            x, y = curvature[:, i]
            dx = ((x - x0) * self.voxel_dim[0]) ** 2
            dy = ((y - y0) * self.voxel_dim[1]) ** 2
            distance[i] = np.sqrt(dx + dy) + distance[i-1]
            x0, y0 = curvature[:, i]

        return torch.from_numpy(curvature.copy()), torch.from_numpy(distance.copy()), torch.from_numpy(apex.copy())

    def fitEPL(self, path: str, pix2um: float = 3.4616):
        """
        Fits a spline through a user defined list of points.

        :param path: Path to a user defined set of points from BASE to APEX as selected in FIJI.
        :return: Tuple[Tensor, Tensor, Tensor]:
            - equal_spaced_points: [2, N] Tensor of pixel locations
            - percentage: [N] Tensor of cochlear percentage
            - apex: Tensor location of apex (guess)
        """

        # csv = np.genfromtxt(path, delimiter=',', skip_header=1)
        csv = pandas.read_csv(path)

        x: Tensor = torch.tensor(csv['X'].to_list())
        y: Tensor = torch.tensor(csv['Y'].to_list())
        center = torch.cat([x.unsqueeze(0), y.unsqueeze(0)]).mean(dim=1)

        x = x - center[0]
        y = y - center[1]

        theta = torch.arctan(y/x)
        r = torch.sqrt(x**2 + y**2)

        loc = torch.abs(theta[0:-2:1] - theta[1:-1:1])

        # if np.abs(loc.max()) > 20 / 180 * np.pi:
        #     theta[loc.argmax()+1::] += 2*np.pi

        tck, u = splprep([theta.numpy(), r.numpy()])#, w=np.ones(len(r)) / len(r), s=1e-2, k=3)
        u_new = np.linspace(0, 1, 1000)
        theta_, r_ = splev(u_new, tck)
        theta_ = torch.from_numpy(theta_)
        r_ = torch.from_numpy(r_)

        x_spline = r_ * torch.cos(theta_) + center[0]
        y_spline = r_ * torch.sin(theta_) + center[1]

        tck, u = splprep([x.numpy(), y.numpy()])#, w=np.ones(len(r)) / len(r), s=1e-2, k=3)
        u_new = np.linspace(0, 1, 1000)
        x_, y_ = splev(u_new, tck)

        x_ = torch.from_numpy(x_) + center[0]
        y_ = torch.from_numpy(y_) + center[1]
        equal_spaced_distance = 0.0001
        curvature  = []
        for i, coord in enumerate(zip(x_, y_)):
            if i == 0:
                base = coord
                curvature .append(base)
            if np.sqrt((base[0] - coord[0]) ** 2 + (base[1] - coord[1]) ** 2) > equal_spaced_distance:
                curvature.append(coord)
                base = coord

        curvature = torch.tensor(curvature).mul(pix2um).T  # <-- mm to px
        # curvature = curvature.T

        # Assume manual is always base to apex
        apex = curvature[:, -1]

        # sort curvature so that it ALWAYS GOES base to apex - UNNESSECARY!
        # curvature = curvature[:, ::-1] if np.all(apex == curvature[:, 0]) else curvature

        total_distance = 0
        distance = torch.zeros(curvature.shape[1])
        x0, y0 = curvature[:, 0]
        # calculate the distance from
        for i in range(1, curvature.shape[1]):
            # each pixel is 288.88nm
            x, y = curvature[:, i]
            dx = ((x - x0) * self.voxel_dim[0]) ** 2
            dy = ((y - y0) * self.voxel_dim[1]) ** 2
            distance[i] = np.sqrt(dx + dy) + distance[i-1]
            x0, y0 = curvature[:, i]

        return curvature, distance, apex

    def _preprocess(self, image: Tensor, diagnostic: Optional[bool] = False):
        """
        Preprocessing of an input image for cochlear path detection.
        Performs downscaling -> binary closing -> diameter closing, -> binary erosion (N times) -> binary dilation (N times) -> Skeletonization

        :param image: Torch tensor image which to preprocess
        :param diagnostic: Displays a matplotlib figure of the image after preprocessing.
        :return:
        """

        image = skimage.transform.downscale_local_mean(image.numpy(), (10, 10)) > 0
        image = skimage.morphology.binary_closing(image)
        image = skimage.morphology.diameter_closing(image, 10)

        if diagnostic:
            plt.imshow(image)
            plt.show()

        for i in range(self.erode):
            image = skimage.morphology.binary_erosion(image)
        for i in range(self.erode):
            image = skimage.morphology.binary_dilation(image)

        if diagnostic:
            plt.imshow(image)
            plt.show()

        return skimage.morphology.skeletonize(image, method='lee')


def merge_regions(destination: Tensor,
                  data: Tensor,
                  threshold: float = 0.25) -> Tuple[Tensor, Tensor]:
    """
    DEPRECIATED! This is legacy code and will be removed.

    assume [C, X, Y, Z] shape for all tensors

    Takes data and puts it in destination intellegently.
    :param destination:
    :param data:
    :return:
    """
    raise DeprecationWarning('Unsure this function is called.')

    expanded = torch.zeros(destination.shape, dtype=destination.dtype, device=data.device)

    # x_pad = (destination.shape[1] - data.shape[1]) // 2
    # y_pad = (destination.shape[2] - data.shape[2]) // 2

    expanded[:, 0:data.shape[1], 0:data.shape[2], 0:data.shape[3]] = data

    # Overlapping pixels will have a value not identical to either when summed
    overlap = destination + expanded
    overlap = torch.logical_not(torch.logical_and(overlap == expanded,
                                                  overlap == destination))

    unique_destination, counts_destination = torch.unique(destination[overlap], return_counts=True)
    unique_data, counts_data = torch.unique(expanded[overlap], return_counts=True)

    # These probably will be small so wont worry about nested loops
    for u_des, c_des in zip(unique_destination, counts_destination):
        if u_des == 0:
            continue
        a = destination == u_des
        for u_data, c_data in zip(unique_data, counts_data):
            if u_data == 0:
                continue
            b = expanded == u_data

            # If iou of each thing is greater than threshold, use branchless programing
            # To make things zero or not
            if _iou(a.float(), b.float()) > threshold:
                destination[a] *= (c_des > c_data)
                expanded[b] *= (c_des < c_data)

    destination[expanded != 0] = expanded[expanded != 0]
    return destination, torch.unique(expanded, return_counts=False)


if __name__ == '__main__':
    pass
