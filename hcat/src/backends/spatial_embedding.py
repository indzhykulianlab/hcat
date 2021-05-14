import torch
import torch.nn as nn

import src.functional
from src.models.r_unet_LTS import r_unet_LTS as RUnetLTS
from src.models.r_unet import embed_model as RUnet
from src.transforms import median_filter, erosion
import src.utils

import matplotlib.pyplot as plt
import skimage.io as io

from typing import Dict, List, Tuple, Optional


class SpatialEmbedding(nn.Module):
    """
    Non scriptable all in one wrapper for spatial embedding
    """

    def __init__(self,
                 sigma: torch.Tensor = torch.tensor([0.02, 0.02, 0.02]),
                 device: str = 'cuda',
                 model_loc: str = None,
                 postprocessing: bool = True,
                 scale: int = 25,
                 figure: Optional[str] = None):

        super(SpatialEmbedding, self).__init__()

        self.scale = scale
        self.device = device
        self.sigma = sigma.to(device)
        self.postprocessing = postprocessing

        self.figure = figure

        self.model = self._model_loader(model_loc, device)

        self.vector_to_embedding = torch.jit.script(
            src.functional.VectorToEmbedding(scale=scale).requires_grad_(False)).eval()
        self.embedding_to_probability = torch.jit.script(
            src.functional.EmbeddingToProbability(scale=scale).requires_grad_(False)).eval()
        self.estimate_centroids = src.functional.EstimateCentroids(scale=scale).requires_grad_(False)

        self.filter = median_filter(kernel_targets=3, rate=1, device=device)
        self.binary_erosion = erosion(device=device)

        self.intensity_rejection = torch.jit.script(src.functional.IntensityCellReject().requires_grad_(False))
        self.nms = src.functional.nms().requires_grad_(False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Inputs an image and outputs a probability mask of everything seen in the image

        :param image:
        :return:
        """
        assert image.ndim == 5
        assert image.shape[1] == 1
        assert image.min() >= -1
        assert image.max() <= 1

        # image = self.filter(image.to(self.device))
        image = image.to(self.device)
        b, c, x, y, z = image.shape

        if self._is_image_bad(image):
            return torch.zeros((b, 0, x, y, z), device=self.device)

        # Evaluate Neural Network Model
        out: torch.Tensor = self.model(image)

        # Assign Outputs
        probability_map = out[:, [-1], ...]
        out = out[:, 0:3:1, ...]

        # Move offset vectors to embedding space
        out: torch.Tensor = self.vector_to_embedding(out)

        # Estimate Centroids from the embedding
        centroids: Dict[str, torch.Tensor] = self.estimate_centroids(out, probability_map)
        # src.utils.plot_embedding(out, centroids)

        # Generate a probability map from the embedding and predicted centroids
        out: torch.Tensor = self.embedding_to_probability(out, centroids, self.sigma)

        # Reject cell masks that overlap or meet min Myo7a criteria
        out: torch.Tesnor = self.intensity_rejection(out, image)

        ind = self.nms(out, 0.5)
        out = out[:, ind, ...]

        # Take probabilities and generate masks!
        probability_map = probability_map.lt(0.8).squeeze(1)
        for i in range(out.shape[1]):
            out[:, i, ...][probability_map] = 0

        self.zero_grad()

        return out

    @staticmethod
    @torch.jit.script
    def _is_image_bad(image: torch.Tensor, min_threshold: float = 0.05):
        """
        Check if an image is likely to NOT contain any cells.

        :param image:
        :param min_threshold:
        :return:
        """
        is_bad = False
        brightness_threshold = torch.tensor(5000).div(2 ** 16).sub(0.5).div(0.5)

        if image.max() == -1:
            is_bad = True
        elif torch.sum(image.gt(brightness_threshold)) < (image.numel() * min_threshold):
            is_bad = True
        return is_bad

    @staticmethod
    def _model_loader(path: str, device):
        try:
            model = torch.jit.script(RUnet(in_channels=1).requires_grad_(False)).to(device)

            if path is not None:
                checkpoint = torch.load(path)
                if isinstance(checkpoint, dict):
                    checkpoint = checkpoint['model_state_dict']
                model.load_state_dict(checkpoint)

        except RuntimeError:  # This is likely due to model weights not lining up.

            model = torch.jit.script(RUnetLTS(1, 4).requires_grad_(False)).to(device)

            if path is not None:
                checkpoint = torch.load(path)
                if isinstance(checkpoint, dict):
                    checkpoint = checkpoint['model_state_dict']
                model.load_state_dict(checkpoint)

        for m in model.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.eval()

        return model


if __name__ == '__main__':
    image = torch.rand((1, 1, 300, 300, 20))
    backend = SpatialEmbedding()
    out = backend(image)
    print(out.shape)
