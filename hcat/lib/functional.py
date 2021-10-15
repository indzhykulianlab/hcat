import pandas
from sklearn.cluster import DBSCAN
import torch
from torch import Tensor
from hcat.train.transforms import erosion
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

from hcat.train.transforms import _crop
from hcat.lib.cell import Cell
from hcat.lib.utils import graceful_exit

import scipy.ndimage
import scipy.ndimage.morphology
from scipy.interpolate import splprep, splev
import GPy

import torchvision.ops
import torch.nn as nn

import matplotlib.pyplot as plt


class VectorToEmbedding(nn.Module):
    def __init__(self, scale: Tensor = torch.tensor([25]), device='cuda'):
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor([scale])
        self.scale = scale.to(device) if scale.numel() != 1 else torch.tensor([scale, scale, scale],
                                                                            device=device).float()
        super(VectorToEmbedding, self).__init__()

    def forward(self, vector: Tensor) -> Tensor:
        """
        Constructs a mesh grid and adds the vector matrix to it.

        Roughly performs the following operation.

        embedding = vector + meshgrid(vector.shape)

        Example:

        >>> from hcat.lib.functional import VectorToEmbedding
        >>> from hcat.models.r_unet import embed_model as model
        >>> import torch
        >>> image = torch.load('image.trch') # [B, C, X, Y, Z]
        >>> vector = model(image)
        >>> vec2emb = torch.jit.script(VectorToEmbedding)
        >>> embedding = vec2emb(vector)


        :param vector: [B, 3=[x,y,z], X, Y, Z] prediction vector from spatial embedding model
        :return: [B, 3=[x,y,z], X, Y, Z] projected spatial embedding vector
        """
        if vector.ndim != 5: raise ShapeError('Expected input tensor ndim == 5')

        num: Tensor = self.scale.float()

        x_factor = 1 / num[0]
        y_factor = 1 / num[1]
        z_factor = 1 / num[2]

        xv, yv, zv = torch.meshgrid(
            [torch.linspace(0, x_factor * vector.shape[2], vector.shape[2], device=vector.device),
             torch.linspace(0, y_factor * vector.shape[3], vector.shape[3], device=vector.device),
             torch.linspace(0, z_factor * vector.shape[4], vector.shape[4], device=vector.device)])

        mesh = torch.cat((xv.unsqueeze(0).unsqueeze(0),
                          yv.unsqueeze(0).unsqueeze(0),
                          zv.unsqueeze(0).unsqueeze(0)), dim=1)

        # Return addition in place if not training
        return mesh + vector if self.training else mesh.add_(vector)


class EmbeddingToProbability(nn.Module):
    def __init__(self, scale: Tensor = torch.tensor([25]), device='cuda'):
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor([scale])
        self.scale = scale.to(device) if scale.numel() != 1 else torch.tensor([scale, scale, scale], device=device)
        super(EmbeddingToProbability, self).__init__()

    def forward(self, embedding: Tensor, centroids: Tensor, sigma: Tensor) -> Tensor:
        """
        Calculates the euclidean distance between the centroid and the embedding
        embedding [B, 3, X, Y, Z] -> euclidean_norm[B, 1, X, Y, Z]
        euclidean_norm = sqrt(Δx^2 + Δy^2 + Δz^2) where Δx = (x_embed - x_centroid_i)

                             /    (e_ix - C_kx)^2       (e_iy - C_ky)^2        (e_iz - C_kz)^2   \
          prob_k(e_i) = exp |-1 * ----------------  -  -----------------   -  ------------------  |
                            \     2*sigma_kx ^2         2*sigma_ky ^2          2 * sigma_kz ^2  /


        Example:

        >>> from hcat.lib.functional import EmbeddingToProbability
        >>> import torch
        >>> embed = torch.load('embed.trch') # [B, C, X, Y, Z]
        >>> emb2prob = torch.jit.script(EmbeddingToProbability)
        >>> probability = emb2prob(embed)


        :param embedding: [B, K=3, X, Y, Z] embedding tensor where K is the likely centroid component: {X, Y, Z}
        :param centroids: [B, N, K_true=3] object centroids where N is the number of instances in the image and K_true is centroid {x, y, z}
        :param sigma: Tensor of shape = (1) or (embedding.shape)
        :return: [B, N, X, Y, Z] of probabilities for instance N
        """
        num = self.scale

        b, _, x, y, z = embedding.shape
        _, n, _ = centroids.shape

        # Centroids might be empty. In this case, return empty array!
        if n == 0:
            return torch.zeros((b, n, x, y, z), device=embedding.device)

        sigma = sigma + torch.tensor([1e-10], device=centroids.device)  # when sigma goes to zero, things tend to break

        if sigma.numel() == 1:
            sigma = torch.cat((sigma, sigma, sigma), dim=0)

        # Sometimes centroids might be in pixel coords instead of scaled.
        # If so, scale by num (usually 512)
        centroids = centroids / num if centroids.max().gt(5) else centroids

        b, _, x, y, z = embedding.shape
        _, n, _ = centroids.shape
        prob = torch.zeros((b, n, x, y, z), device=embedding.device)

        # Common operation. Done outside of loop for speed.
        sigma = sigma.pow(2).mul(2)

        # Calculate euclidean distance between centroid and embedding for each pixel and
        # turn that distance to probability and put it in preallocated matrix for each n
        # In eval mode uses in place operations to save memory!

        if self.training:
            for i in range(n):
                euclidean_norm = (embedding - centroids[:, i, :].view(centroids.shape[0], 3, 1, 1, 1)).pow(2)
                prob[:, i, :, :, :] = torch.exp(
                    (euclidean_norm / sigma.view(centroids.shape[0], 3, 1, 1, 1)).mul(-1).sum(dim=1)).squeeze(1)

        else:  # If in Eval Mode - do a bunch of inplace operations for speed and memory efficiency
            for i in range(centroids.shape[1]):
                euclidean_norm = (embedding - centroids[:, i, :].view(centroids.shape[0], 3, 1, 1, 1)).pow(2)
                prob[:, i, :, :, :] = torch.exp(
                    euclidean_norm.div_(sigma.view(centroids.shape[0], 3, 1, 1, 1)).mul_(-1).sum(dim=1)).squeeze(1)
        return prob


@torch.jit.script
def learnable_centroid(embedding: Tensor, mask: Tensor, method: str = 'mean',
                       scale: Tensor = torch.tensor([25])) -> Tensor:
    """
    Estimates a center of attraction best predicted by the network.

    Take mean direction of vectors predicted for a single object and takes that as the centroid of the object.
    In this way the model can learn where best to predict a center.

    Example:

    >>> from hcat.lib.functional import learnable_centroid
    >>> import torch
    >>> embed = torch.load('embed.trch') # [B, C, X, Y, Z] predicted embedding
    >>> mask = torch.load('mask.trch') # [B, C, X, Y, Z] ground truth mask
    >>> centroids = learnable_centroid(embed)

    :param embedding: [B, C, X, Y, Z] predicted embedding tensor from deep learning model
    :param mask: [B, C, X, Y, Z] ground truth mask
    :param method: ['mode' or 'mean'] method by which to calculate centroids. Either can calculate the centroid of via "median" or "mean" of predicted embeddings
    :return: [C, N, k=3] centroids for N objects
    """
    with torch.no_grad():
        if method != 'mean' and method != 'mode':
            raise NotImplementedError(f'Cannot estimate centroids with method {method}')

        shape = embedding.shape
        mask = _crop(mask, 0, 0, 0, shape[2], shape[3], shape[4])  # from hcat.transforms._crop
        centroid = torch.zeros((mask.shape[0], mask.shape[1], 3), device=mask.device)

        # Loop over each instance and take the mean of the vectors multiplied by the mask (0 or 1)
        for i in range(mask.shape[1]):
            ind = torch.nonzero(embedding * mask[:, i, ...].unsqueeze(1))  # [:, [B, N, X, Y, Z]]
            center = torch.tensor([-10, -10, -10], device=mask.device)

            # Guess Centroids by average location of all vectors or just by the most common value (mean vs mode).
            if method == 'mean' and ind.shape[0] > 1:
                center = embedding[ind[:, 0], :, ind[:, 2], ind[:, 3], ind[:, 4]].mean(0)
            elif method == 'mode' and ind.shape[0] > 1:
                center = embedding[ind[:, 0], :, ind[:, 2], ind[:, 3], ind[:, 4]].mul(scale).round().mode(0)[0].div(scale)  # mode gives vals and indicies

            centroid[:, i, :] = center

        return centroid * scale.to(centroid.device)


class EstimateCentroids(nn.Module):
    def __init__(self, downsample: int = 1,
                 n_erode: int = 4,
                 eps: float = 1,
                 min_samples: int = 10,
                 scale: Tensor = torch.tensor([25]),
                 device='cuda'):
        """
        Estimates centroids of objects in a spatial embedding matrix.

        :param downsample: factor by which to downsample embedding matrix, higher numbers increase computational speed.
        :param n_erode: Erodes probability map, higher numbers prefentiall select voxels near object centers.
        :param eps: tuning parameter for DBSCAN
        :param min_samples: minimum number of voxels for an image to be detected
        :param scale: scaling factor of vector embedding. Must be identical to VectorToEmbedding
        :param device: physical device by which to perform computation on.
        """

        super(EstimateCentroids, self).__init__()
        self.downsample = downsample
        self.n_erode = n_erode
        self.eps = eps
        self.min_samples = min_samples
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor([scale])
        self.scale = scale.to(device) if scale.numel != 1 else torch.tensor([scale, scale, scale], device=device)

    def forward(self, embedding: Tensor, probability_map: Tensor = None,
                ) -> Tensor:
        """
        Estimates centroids of objects in a spatial embedding matrix.

        Employs DBSCAN clustering algorithm to predict embedding clusters in a predicted embedding matrix

        Example:

        >>> from hcat.lib.functional import EstimateCentroids
        >>> import torch
        >>> embed = torch.load('embed.trch') # [B, C, X, Y, Z] predicted embedding
        >>> mask = torch.load('mask.trch') # [B, C, X, Y, Z] ground truth mask
        >>> estimate = EstimateCentroids()
        >>> centroids = estimate(embed, mask)

        :param embedding: [B, C=3, X, Y, Z] predicted spatial embedding matrix
        :param probability_map: [B, C=1, X, Y, Z] predicted object probability mask
        :return: [B, N, K=3] Predicted centroids
        """
        num = self.scale

        device = embedding.device
        embedding = embedding.squeeze(0).reshape((3, -1)).clone()
        binary_erosion = erosion(device=device)

        # First Step is to prune number of viable points
        # this is done optionally by passing the probability map of the image
        # Runs a binary erosion to only consider pixels in the middle of cells
        if probability_map is not None:
            if probability_map.ndim != 5: raise ShapeError(f'probability_map.ndim != 5, {probability_map.ndim}')
            if probability_map.shape[0] != 1: raise ShapeError(f'Batch size should be 1, not {probability_map.shape}')
            if probability_map.shape[1] != 1: raise ShapeError(f'Color size should be 1, not {probability_map.shape}')

            pm = probability_map.squeeze(0).gt(0.5).float()  # Needs 4 channel input. Remove Batch.
            for i in range(self.n_erode):
                pm = binary_erosion(pm)

            embedding[:, torch.logical_not(pm.squeeze(0).gt(0.5).view(-1))] = -10

        x = embedding[0, :]
        y = embedding[1, :]
        z = embedding[2, :]

        ind_x = x > -2
        ind_y = y > -2
        ind_z = z > -2
        ind = torch.logical_or(ind_x, ind_y)
        ind = torch.logical_or(ind, ind_z)
        embedding = embedding[:, ind]

        embedding = embedding[:, 0:-1:self.downsample].mul(num).round().detach().cpu()

        if embedding.shape[-1] > 3_000_000:
            raise RuntimeError(f'Unsuported number of samples. {embedding.shape[-1]} after pruning, 3,000,000 maximum.')

        if embedding.shape[1] == 0:
            return torch.empty((1, 0, 3), device=device)

        db = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=-1).fit(embedding.numpy().T)
        labels = torch.from_numpy(db.labels_)

        unique, counts = torch.unique(labels, return_counts=True)
        centroids = torch.zeros((1, 3, unique.shape[0]), device=device)  # [B, C=3, N]
        scores = torch.zeros((1, 1, unique.shape[0]), device=device)  # [B, C=1, N]

        # Assign mean value of labels to the centroid
        for i, (u, s) in enumerate(zip(unique, counts)):
            if u == -1:
                continue
            centroids[0, :, i] = embedding[:, labels == u].mean(axis=-1)
            scores[0, 0, i] = s

        if centroids.sum() == 0:  # nothing detected
            return torch.empty((1, 0, 3), device=device)

        # Works best with non maximum supression

        centroids_xy = centroids[0, [0, 1], :].T
        wh = torch.ones(centroids_xy.shape,
                        device=device) * 45  # <- I dont know why this works but it does so deal with it????
        boxes = torch.cat((centroids_xy, wh), dim=-1)
        boxes = torchvision.ops.box_convert(boxes, 'cxcywh', 'xyxy')
        if boxes.ndim > 2: boxes = boxes.squeeze(0)
        if boxes.shape[0] > 1:
            keep = torchvision.ops.nms(boxes.squeeze(0), scores.squeeze().float(), 0.5)
        else:
            keep = torch.ones(centroids.shape[-1]).gt(0)

        return centroids[:, :, keep].transpose(1, 2)


@torch.jit.script
def _iou(a: Tensor, b: Tensor) -> Tensor:
    """
    Takes two identically sized tensors and computes IOU of the Masks
    Assume a, b are boolean tensors with values 0 and 1

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
        >>> nms = torch.jit.script(nms) # [1, 20, 100, 100, 10]
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


# @graceful_exit('\x1b[1;31;40m' + 'ERROR: Could not calculate cochlear length.' + '\x1b[0m')
class PredictCurvature:
    def __init__(self,
                 voxel_dim=(.28888, .28888, .28888 * 3), # um
                 equal_spaced_distance: float = 0.01,
                 erode: int = 3,
                 scale_factor=10,
                 method=None):

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
        Try's various methods to predict curvature.
        If user provides a csv of curvature default to this.
        In order -> CSV, Myo7a, Mask

        :param image:
        :param mask:
        :param csv:
        :return:
        """
        curvature, distance, apex = None, None, None
        if self.method is None:
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

        :param path:
        :return:
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

    def _preprocess(self, image, diagnostic):

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



########################################################################################################################
#                                                 U Net
########################################################################################################################

class PredictCellCandidates(nn.Module):
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Initializes a Faster RCNN model for detection of cell candidates.
        These cell candidates are used as seed values for a watershed segmentation algorithm.

        :param model: Initialized faster RCNN model
        :param device: device used to perform calculations ('cpu' or 'cuda')
        """
        super(PredictCellCandidates, self).__init__()
        self.model = model.to(device)
        self.device = device

    def forward(self, image: Tensor) -> Dict[str, Tensor]:
        """
        Applies a FasterRCNN model on each z plane of an image. Returns the prediction in a dictionary.

        Example:

        >>> from hcat.lib.functional import PredictCellCandidates
        >>> import torch
        >>> image = torch.load('image.trch') # [B, C, X, Y, Z] ground truth mask
        >>> predict = PredictCellCandidates()
        >>> centroids: Dict[str, Tensor] = predict(image) # keys: 'scores', 'boxes', 'labels'

        :param image: [B, C=4, X, Y, Z] input image
        :return: a dictionary with keys: 'scores', 'boxes', 'labels'.
        """

        # Check Inputs
        if not isinstance(image, Tensor): raise ValueError(f'Image should be torch.tensor {type(image)}')
        if image.ndim != 5: raise ValueError(f'Image dimmensions should be 5, not {image.ndim}')
        if image.shape[0] != 1: raise ValueError(
            f'Multiple Batches not supported. Image.shape[0] should be 1, not {image.shape[0]}')

        max_num_cells = 0
        out = {'boxes': torch.tensor([]), 'scores': torch.tensor([]), 'labels': torch.tensor([])}

        with torch.no_grad():
            for z in range(image.shape[-1]):  # ASSUME TORCH VALID IMAGES [B, C, X, Y, Z]

                # Take a mini chunk out of the original image
                image_z_slice = image[..., z]

                # Apply faster rcnn prediction model to this
                # We're only doing a batch size of 1, take the first dict of results
                # list of dicts (batch size) with each dict containing 'boxes' 'labels' 'scores'
                predicted_cell_locations = self.model(image_z_slice.to(self.device))[0]

                # We need to know what z plane these cells were predicted, add this as a new index to the dict
                z_level = torch.ones(len(predicted_cell_locations['scores'])) * z
                predicted_cell_locations['z_level'] = z_level

                if len(predicted_cell_locations['scores']) > len(out['scores']):
                    out = predicted_cell_locations

            # Move everything to be a float and on the cpu()
            for name in out:
                out[name] = out[name].to(self.device).float()

        return out


class PredictSemanticMask(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 device: str = 'cpu',
                 use_probability_map: bool = False,
                 mask_cell_prob_threshold: float = 0.5):
        super(PredictSemanticMask, self).__init__()

        self.device = device
        self.use_prob_map = use_probability_map
        self.mask_threshold = mask_cell_prob_threshold

        self.unet = model.to(device)

    def forward(self, image: Tensor) -> Tensor:
        """
        Uses pretrained unet model to predict semantic segmentation of hair cells.

        ALGORITHM:
        Remove inf and nan ->
        apply padding ->
        calculate indexes for unet based on image.shape ->
        apply Unet on slices of image based on indexes ->
        take only valid portion of the middle of each output mask ->
        construct full valid mask ->
        RETURN mask


        BLUR PROBABILITY MAP???

        :param unet: Trained Unet Model from hcat.unet
        :param image: Tensor image with transforms pre applied!!! [1, 4, X, Y, Z]
        :param device: 'cuda' or 'cpu'
        :param mask_cell_prob_threshold: float between 0 and 1 : min probability of cell used to generate the mask
        :param use_probability_map: bool, if True, does not apply sigmoid function to valid out, returns prob map instead
        :return mask:

        """

        # Check Inputs
        if not isinstance(image, Tensor): raise ValueError(f'Image should be torch.tensor {type(image)}')
        if image.ndim != 5: raise ValueError(f'Image dimmensions should be 5, not {image.ndim}')
        if image.shape[0] != 1: raise ValueError(
            f'Multiple Batches not supported. Image.shape[0] should be 1, not {image.shape[0]}')

        PAD_SIZE = (30, 30, 4)

        # inf and nan screw up model evaluation. Happens occasionally. Remove them!
        image[torch.isnan(image)] = 0
        image[torch.isinf(image)] = 1

        im_shape = image.shape

        # Apply Padding by reflection to all edges of image
        # With pad size of (30, 30, 4)
        # im.shape = [1,4,100,100,10] -> [1, 4, 160, 160, 18]
        image = hcat.lib.utils.pad_image_with_reflections(torch.as_tensor(image), pad_size=PAD_SIZE)

        with torch.no_grad():

            valid_out = self.unet(image.to(self.device))

            # Unet cuts off bit of the image. It is unavoidable so we add padding to either side
            # We need to remove padding from the output of unet to get a "Valid" segmentation
            valid_out = valid_out[:, :,
                        PAD_SIZE[0]:im_shape[2] + PAD_SIZE[0],
                        PAD_SIZE[1]:im_shape[3] + PAD_SIZE[1],
                        PAD_SIZE[2]:im_shape[4] + PAD_SIZE[2]]

            # Take pixels that are greater than 50% likely to be a cell.
            if not self.use_prob_map:
                valid_out.gt_(self.mask_threshold)  # Greater Than
                valid_out = valid_out.type(torch.uint8)

        return valid_out


class GenerateSeedMap(nn.Module):
    def __init__(self,
                 cell_prob_threshold: float = 0.5,
                 mask_prob_threshold: float = 0.5):
        super(GenerateSeedMap, self).__init__()

        self.cell_threshold = cell_prob_threshold
        self.mask_threshold = mask_prob_threshold

    def forward(self, cell_candidates: Dict[str, Tensor], prob_map: Tensor) -> Tensor:
        """

        :param cell_candidates: Dict[str, Tensor] keys = ['scores', 'boxes', 'z_plane', 'labels']
        :param prob_map: [B, C, X, Y, Z] Tensor
        :return:
        """

        unique_cell_id = 2  # 1 is reserved for background
        expand_z = 4  # dialate the z to try and account for nonisotropic zstacks
        b, c, x, y, z = prob_map.shape
        seed = torch.zeros(prob_map.shape, dtype=np.int)

        if prob_map.max() == 0 or len(cell_candidates['scores']) == 0:
            return seed

        z = cell_candidates['z_level'].cpu()
        scores = cell_candidates['scores'].cpu()
        boxes = cell_candidates['boxes']

        z = z[scores > self.cell_threshold]
        prob = scores[scores > self.cell_threshold]

        if prob.sum() == 0:
            return torch.zeros(prob_map.shape)

        boxes = boxes[scores > self.cell_threshold, ...]

        # Try to basically remove boxes that arent likely to be over a cell.
        x1, y1, x2, y2 = boxes.cpu().T
        centers = torch.cat((torch.round(x1 + (x2 - x1)).unsqueeze(0),
                             torch.round(y1 + (y2 - y1)).unsqueeze(0),
                             z.unsqueeze(0)
                             ), dim=0)

        # Loop over every center and see if it lies over a region of high probability on the probability map
        # This prunes cell candidates to reasonable ones
        ind = torch.zeros(len(x1))
        for i, (x0, y0, z0) in enumerate(centers.T):
            try:
                if prob_map[0, 0, int(x0), int(y0), int(z0)] > 0.5:
                    ind[i] = 1
            except Exception:
                pass

        # We look at the cell candidates z planes and choose just one that is optimal
        # This is done by looking at the larges z plane with the biggest mean score.
        unique_z, counts = torch.unique(z, return_counts=True)
        best_z = 0
        best_z_avg = 0
        for uni in unique_z:
            if prob[z == uni].mean() > best_z_avg:
                best_z = uni
                best_z_avg = prob[z == uni].mean()

        # Sort to put close centers near each other
        sorted, indices = torch.sort(boxes, dim=0)
        for key in cell_candidates:
            cell_candidates[key] = cell_candidates[key][indices[:, 0], ...]

        # Generate Seed Map
        for i, (y1, x1, y2, x2) in enumerate(cell_candidates['boxes']):

            # There are various conditions where a box is invalid
            # in these cases, do not place a seed and skip the box
            if x1 > prob_map.shape[2]:
                continue  # box is outside x dim
            elif y1 > prob_map.shape[3]:
                continue  # box is outside y dim
            elif cell_candidates['scores'][i] < self.cell_threshold:
                continue  # box probability is lower than predefined threshold
            elif cell_candidates['z_level'][i] < best_z - 2:
                continue  # box is on the wrong z plane within a tolerance
            elif cell_candidates['z_level'][i] > best_z + 2:
                continue  # box is on the wrong z plane within a tolerance

            # in the cases where a box is clipping the outside, crop it to the edges
            if x2 > prob_map.shape[2]:
                x2 = torch.tensor(prob_map.shape[2] - 1).float()
            elif y2 > prob_map.shape[3]:
                y2 = torch.tensor(prob_map.shape[3] - 1).float()

            # Each box is a little to conservative in its estimation of a hair cell
            # To compensate, we add dx and dy to the corners to increase the size
            dx = [5, -5]
            dy = [5, -5]

            # Check if adding will push box outside of range!
            if (x1 + dx[0]) < 0:
                dx[0] = x1
            if (y1 + dy[0]) < 0:
                dy[0] = y1
            if (x2 + dx[1]) > prob_map.shape[2]:
                dx[1] = prob_map.shape[2] - x2
            if (y2 + dy[1]) > prob_map.shape[3]:
                dy[1] = prob_map.shape[3] - y2

            x1 = int(x1.clone().add(dx[0]).round())
            x2 = int(x2.clone().add(dx[1]).round())
            y1 = int(y1.clone().add(dy[0]).round())
            y2 = int(y2.clone().add(dy[1]).round())

            box = prob_map[0, 0, int(x1):int(x2), int(y1):int(y2), int(best_z)]

            # Here we place a seed value for watershed at each point of the valid box
            seed_square = 2
            for i in range(seed_square):
                if i + best_z > seed.shape[-1]:
                    continue
                try:
                    seed[0, 0, int(x1):int(x2), int(y1):int(y2), int(best_z) + i][box == box.max()] = int(
                        unique_cell_id)
                except IndexError:
                    if seed.max() > 0:
                        continue
                    else:
                        raise IndexError(f'No Seed was placed when index')
                except ValueError:
                    continue
            unique_cell_id += 1

        return seed


class InstanceMaskFromProb(nn.Module):
    def __init__(self,
                 mask_prob_threshold: float = 0.5,
                 expand_z: int = 5,
                 use_prob_map: bool = False):
        super(InstanceMaskFromProb, self).__init__()

        self.mask_threshold = mask_prob_threshold
        self.use_prob_map = use_prob_map
        self.expand_z = expand_z

    def forward(self, semantic_mask: Tensor,
                seed: Tensor) -> Tensor:
        """

        :param semantic_mask:
        :param seed:
        :return:
        """

        if seed.max() == 0:
            b, c, x, y, z = seed.shape
            return torch.zeros((b, 0, x, y, z))

        assert semantic_mask.shape == seed.shape

        # Preallocate some matrices
        mask = semantic_mask.gt(self.mask_threshold)

        # Context aware probability map
        # If true, watershed is done on the probability map, distance is now probability
        # If false, run a distance transform on the binary mask and run watershed on that
        base = self._prepare_watershed_base(semantic_mask)

        base = self._expand_z_dimmension(base, self.expand_z)
        seed = self._expand_z_dimmension(seed, self.expand_z)
        mask = self._expand_z_dimmension(mask, self.expand_z)

        assert mask.shape == seed.shape
        assert base.shape == mask.shape

        # Run the watershed algorithm
        mask = skimage.segmentation.watershed(image=base.cpu().numpy() * -1,
                                              markers=seed.cpu().numpy(),
                                              mask=mask.cpu().numpy(),
                                              connectivity=1,
                                              watershed_line=True, compactness=0.01)

        mask = torch.from_numpy(mask)
        mask = self._correct_predicted_instance_mask(mask)

        return mask.unsqueeze(0).unsqueeze(0)

    def _prepare_watershed_base(self, semantic_mask: Tensor):
        """
        Prepare the base matrix which watershed is run off of. Has two potential methods depending on context.
            1: use distance transform of semantic segmentation mask
            2: use the probability map predicted by the algorithm

        :param semantic_mask:
        :return:
        """
        distance = torch.zeros(semantic_mask.shape)
        if self.use_prob_map and semantic_mask.max() > 1:
            semantic_mask.add_(1e-8).sub_(torch.min(semantic_mask)).div_(torch.max(semantic_mask))
            distance[0, 0, :, :, :] = semantic_mask[0, 0, :, :, :]
        else:
            for z in range(distance.shape[-1]):
                distance[0, 0, :, :, z] = torch.from_numpy(
                    scipy.ndimage.morphology.distance_transform_edt(
                        semantic_mask[0, 0, :, :, z].cpu().numpy().astype(np.uint8)))
        return distance

    @staticmethod
    def _expand_z_dimmension(matrix, expand_z):
        """
        Accounts for anisotropy by expanding each z plane of a matrix with copies
        example
        matrix: Tensor = torch.rand([1, 1, 100, 100, 10])
        matrix: Tensor = _expand_z_dimmension(matrix, expand_z = 4)
        matrix.shape
            >>> torch.Shape([1, 100, 100, 40])
        torch.all(matrix[0,:,:,1] == matrix[0,:,:,2])
            >>> True


        :param matrix: Tensor with 5 dimmensions
        :param expand_z: integer multiple
        :return:
        """
        if matrix.ndim != 5: raise ValueError(f'Expected matrix.ndim == 5 matrix not {matrix.ndim}.')
        mat_expanded = torch.zeros((matrix.shape[2], matrix.shape[3], matrix.shape[4] * expand_z))
        for i in range(matrix.shape[4]):
            for j in range(expand_z):
                mat_expanded[:, :, (expand_z * i + j)] = matrix[0, 0, :, :, i]
        return mat_expanded

    def _correct_predicted_instance_mask(self, mask: Tensor) -> Tensor:
        mask[mask == 1] = 0
        mask = mask[..., ::self.expand_z]
        return mask


########################################################################################################################
#                                               Cell Rejection
########################################################################################################################

class IntensityCellReject(nn.Module):
    def __init__(self, threshold: float = 0.08):
        """
        :param threshold: If cell avg val is lower than this, get rid of it...
        """
        super(IntensityCellReject, self).__init__()
        self.threshold = threshold

    def forward(self, mask: Tensor, image: Tensor) -> Tensor:
        """
        Checks the type of mask: colormask or binary mask.
        :param mask:
        :param image:
        :return:
        """
        mask, image = hcat.lib.utils.crop_to_identical_size(mask, image)

        # strange edge case where

        if mask.shape[1] == 1 and mask.max() >= 1:
            return self.color(mask, image)
        elif mask.shape[1] >= 1 and mask.max() < 1:
            return self.binary(mask, image)
        elif mask.shape[1] == 0:
            return torch.empty((0)).gt(0)
        else:
            raise RuntimeError(f'Unknown Mask Shape: {mask.shape}, {mask.max()}')

    def color(self, mask: Tensor, image: Tensor) -> Tensor:
        ind = torch.ones(mask.shape[1], dtype=torch.long, device=mask.device)

        unique = torch.unique(mask)
        for u in unique:
            if u == 0:
                continue
            m = (mask == u)

            if (m * image).sum().div(m.sum()) < ((self.threshold - 0.5) / 0.5):
                mask[m] = 0

        return mask

    def binary(self, mask: Tensor, image: Tensor) -> Tensor:
        """
        Re
        :param mask: [B, N, X, Y, Z]
        :param image: [B, 1, X, Y, Z]
        :return:
        """

        mask, image = hcat.lib.utils.crop_to_identical_size(mask, image)
        ind = torch.ones(mask.shape[1], dtype=torch.long, device=mask.device)

        for i in range(mask.shape[1]):
            m = mask[:, i, ...].unsqueeze(1).gt(0.5)
            if (m * image).sum().div(m.sum()) < ((self.threshold - 0.5) / 0.5):
                ind[i] = 0

        return mask[:, ind.gt(0), ...]


# @graceful_exit('\x1b[1;31;40m' + 'ERROR: Could not merge regions.' + '\x1b[0m')
# @torch.jit.script
def merge_regions(destination: Tensor,
                  data: Tensor,
                  threshold: float = 0.25) -> Tuple[Tensor, Tensor]:
    """
    assume [C, X, Y, Z] shape for all tensors

    Takes data and puts it in destination intellegently.
    :param destination:
    :param data:
    :return:
    """
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
