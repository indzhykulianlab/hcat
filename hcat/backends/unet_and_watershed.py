import torch
import torch.nn as nn
from typing import Dict, Optional

import hcat.lib.functional
from hcat.models.r_unet import wtrshd_model as WatershedUnet
from hcat.models.depreciated.rcnn import rcnn


# DOCUMENTED

class UNetWatershed(nn.Module):
    def __init__(self,
                 device: str = 'cuda',
                 unet_model_path: str = None,
                 frcnn_model_path: str = None):
        """
        Initalize UNet + Watershed Segmentation Algorithm

        :param device: device to run analysis on.
        :param unet_model_path: path to trained unet
        :param frcnn_model_path: path to trained fasterrcnn model
        """

        super(UNetWatershed, self).__init__()

        self._in_channels = 1
        self.device = device

        self.model = torch.jit.script(WatershedUnet(in_channels=self._in_channels)).to(device)
        self.ffcnn = rcnn().to(device).eval()

        self.predict_cell_candidates = hcat.lib.functional.PredictCellCandidates(self.ffcnn, device=device)
        self.predict_segmentation_mask = hcat.lib.functional.PredictSemanticMask(model=self.model, device=device)
        self.generate_seed_map = hcat.lib.functional.GenerateSeedMap()

        self.instance_mask_from_prob = hcat.lib.functional.InstanceMaskFromProb()

    # @graceful_exit('\x1b[1;31;40m' + 'ERROR: Unet+Watershed Failed. Aborting...' + '\x1b[0m')
    def forward(self, image: torch.Tensor, channel: Optional[int] = 2) -> torch.Tensor:
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
        if image.shape[1] != 4: raise ValueError('Need 4D image for unet+watershed segmentation.')
        b, c, x, y, z = image.shape

        if self.image_reject and self._is_image_bad(image):
            return torch.zeros((b, 0, x, y, z), device=self.device)

        image_frcnn = image[:, [0, 2, 3], :, :, :]

        cell_candidates: Dict[str, torch.Tensor] = self.predict_cell_candidates(image_frcnn)
        semantic_mask: torch.Tensor = self.predict_segmentation_mask(image[:, [channel], ...])
        seed: torch.Tensor = self.generate_seed_map(cell_candidates, semantic_mask)
        instance_mask: torch.Tensor = self.instance_mask_from_prob(semantic_mask, seed)

        return self._colormask_to_mask(instance_mask)

    def load(self, unet_path: str, fasterrcnn_path: str) -> None:
        """
        Assigns pretrained weights to both necessary backend ML models: unet and fasterrcnn

        :param unet_path: path to pretrained unet model
        :param fasterrcnn_path: path to pretrained faster rcnn model
        :return:
        """

        checkpoint = torch.load(unet_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.ffcnn = rcnn(path=fasterrcnn_path).to(self.device).eval()