import torch
import hcat.lib.functional
from hcat.lib.functional import IntensityCellReject
from hcat.backends.backend import Backend
from hcat.models.r_unet import embed_model as RUnet
from hcat.train.transforms import median_filter, erosion
import hcat.lib.utils
from hcat.lib.utils import graceful_exit

import os.path
import wget

from typing import Dict, Optional


class SpatialEmbedding(Backend):
    def __init__(self,
                 sigma: Optional[torch.Tensor] = torch.tensor([0.02, 0.02, 0.02]),
                 device: Optional[str] = 'cuda',
                 model_loc: Optional[str] = None,
                 postprocessing: Optional[bool] = True,
                 scale: Optional[int] = 25,
                 figure: Optional[str] = None,
                 archetecture: Optional[RUnet] = RUnet):
        """
        Initialize Spatial embedding Algorithm.

        :param sigma: torch.Tensor[sigma_x, sigma_y, sigma_z] values for gaussian probability estimation.
        :param device: String value for torch device by which to run segmentation backbone on.
        :param model_loc: Path to trained model files.
        :param postprocessing: Disable segmentation postprocessing, namely
        :param scale: scale factor based on max diameter of object
        :param figure: filename and path of diagnostic figure which may be rendered
        """

        super(SpatialEmbedding, self).__init__()

        self.url = 'https://github.com/buswinka/hcat/blob/master/modelfiles/spatial_embedding.trch?raw=true'
        # self.url = None
        self.scale = torch.tensor(scale)
        self.device = device
        self.sigma = sigma.to(device)
        self.postprocessing = postprocessing

        self.figure = figure

        if self.url:
            self.model = self._model_loader_url(self.url, archetecture, device)
        else:
            self.model = self._model_loader_path(model_loc, archetecture, device)

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

    @graceful_exit('\x1b[1;31;40m' + 'ERROR: Spatial Embedding Failed. Aborting...' + '\x1b[0m')
    def forward(self, image: torch.Tensor) -> torch.Tensor:
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


        # Evaluate Neural Network Model

        out: torch.Tensor = self.model(image)

        # Assign Outputs
        probability_map = out[:, [-1], ...]
        out = out[:, 0:3:1, ...]

        self.prob = probability_map.cpu()
        self.vec = out.cpu()

        out: torch.Tensor = self.vector_to_embedding(out)

        self.embed = out.cpu()

        centroids: Dict[str, torch.Tensor] = self.estimate_centroids(out, probability_map)

        self.centroids = centroids




        out: torch.Tensor = self.embedding_to_probability(out, centroids, self.sigma)


        # Reject cell masks that overlap or meet min Myo7a criteria
        if self.postprocessing:
            out: torch.Tensor = self.intensity_rejection(out, image)

        # print(centroids.shape, out.shape)

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

    def load(self, model_loc: str) -> None:
        """
        Initializes model weights from a url or filepath.

        Example:

        >>> from hcat.backends.spatial_embedding import SpatialEmbedding
        >>> backend = SpatialEmbedding()
        >>>
        >>> url = 'https://www.model_location.com/model.trch'
        >>> backend.load(url) # Works with url
        >>>
        >>> model_path = 'path/to/my/model.trch'
        >>> backend.load(model_path) # Also works with path


        :param model_loc: url or filepath
        :return: None
        """
        if self._is_url(model_loc):
            return self._model_loader_url(model_loc, RUnet(in_channels=1).requires_grad_(False), self.device)
        else:
            return self._model_loader_path(model_loc, RUnet(in_channels=1).requires_grad_(False), self.device)


