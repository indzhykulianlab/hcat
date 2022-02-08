import torch
from torch import Tensor

import hcat.lib.functional
from hcat.lib.functional import IntensityCellReject
from hcat.backends.backend import Backend
from hcat.models.r_unet import embed_model as RUnet
from hcat.train.transforms import median_filter, erosion
import hcat.lib.utils

from typing import Dict, Optional


class SpatialEmbedding(Backend):
    def __init__(self,
                 sigma: Optional[Tensor] = torch.tensor([0.02, 0.02, 0.02]),
                 device: Optional[str] = 'cuda',
                 model_loc: Optional[str] = None,
                 postprocessing: Optional[bool] = True,
                 scale: Optional[int] = 25,
                 figure: Optional[str] = None,
                 architecture: Optional[RUnet] = RUnet):
        """
        Initialize Spatial embedding Algorithm.

        :param sigma: Tensor[sigma_x, sigma_y, sigma_z] values for gaussian probability estimation.
        :param device: String value for torch device by which to run segmentation backbone on.
        :param model_loc: Path to trained model files.
        :param postprocessing: Disable segmentation postprocessing, namely
        :param scale: scale factor based on max diameter of object
        :param figure: filename and path of diagnostic figure which may be rendered
        """

        super(SpatialEmbedding, self).__init__()

        # self.url = 'https://github.com/buswinka/hcat/blob/master/modelfiles/spatial_embedding.trch?raw=true'
        self.url = None
        self.scale = torch.tensor(scale)
        self.device = device
        self.sigma = sigma.to(device)
        self.postprocessing = postprocessing

        self.figure = figure

        if self.url:
            self.model = self._model_loader_url(self.url, architecture, device)
        else:
            self.model = self._model_loader_path(model_loc, architecture, device)

        self.vector_to_embedding = torch.jit.script(
            hcat.lib.functional.VectorToEmbedding(scale=self.scale).requires_grad_(False).eval())

        self.embedding_to_probability = torch.jit.script(
            hcat.lib.functional.EmbeddingToProbability(scale=self.scale).requires_grad_(False).eval())

        self.estimate_centroids = hcat.lib.functional.EstimateCentroids(scale=self.scale).requires_grad_(False)

        self.filter = median_filter(kernel_targets=3, rate=1, device=device)
        self.binary_erosion = erosion(device=device)

        self.intensity_rejection = IntensityCellReject()

        self.nms = hcat.lib.functional.nms().requires_grad_(False)

        self.centroids = None
        self.vec = None
        self.embed = None
        self.prob = None

    def forward(self, image: Tensor) -> Tensor:
        """
        Inputs an image and outputs a probability mask of everything seen in the image.

        .. note::
           Call the module as a function to execute this method (similar to torch.nn.module).

        .. warning:
           Will not raise an error upon failure, instead returns None and prints to standard out

        Example:

        >>> from hcat.backends.spatial_embedding import SpatialEmbedding
        >>> import torch
        >>> backend = SpatialEmbedding()
        >>> image = torch.load('path/to/my/image.trch')
        >>> assert image.ndim == 5 # Shape should be [B, C, X, Y, Z]
        >>> masks = backend(image)

        :param image: [B, C=4, X, Y, Z] input image
        :return: [B, 1, X, Y, Z] output segmentation mask where each pixel value is a cell id (0 is background)
        """
        assert image.ndim == 5
        assert image.shape[1] == 1
        assert image.min() >= -1
        assert image.max() <= 1

        # image = self.filter(image.to(self.device))
        image = image.to(self.device)
        b, c, x, y, z = image.shape

        if self.image_reject and self._is_image_bad(image):
            return torch.zeros((b, 0, x, y, z), device=self.device)


        out: Tensor = self.model(image) # Evaluate Neural Network Model

        # Assign Outputs
        probability_map = out[:, [-1], ...]
        out = out[:, 0:3:1, ...]

        self.prob = probability_map.cpu()
        self.vec = out.cpu()

        out: Tensor = self.vector_to_embedding(out)
        self.embed = out.cpu()

        self.centroids: Dict[str, Tensor] = self.estimate_centroids(
            self.vector_to_embedding(self.vec.cuda(), n=5), probability_map)

        out: Tensor = self.embedding_to_probability(out, self.centroids, self.sigma)

        # Reject cell masks that overlap or meet min Myo7a criteria
        if self.postprocessing:
            out: Tensor = self.intensity_rejection(out, image)

        if out.numel() == 0:
            return torch.zeros((b, 0, x, y, z), device=self.device)

        ind = self.nms(out, 0.5)
        out = out[:, ind, ...]

        # Take probabilities and generate masks!
        probability_map = probability_map.lt(0.8).squeeze(1)
        for i in range(out.shape[1]):
            out[:, i, ...][probability_map] = 0

        self.zero_grad()

        return out
