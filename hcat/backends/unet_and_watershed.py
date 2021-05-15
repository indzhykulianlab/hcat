import torch
import torch.nn as nn
from typing import Dict, Optional

import hcat.functional as functional
from hcat.models.r_unet import wtrshd_model as WatershedUnet
from hcat.models.rcnn import rcnn


class UNetWatershed(nn.Module):
    def __init__(self,
                 device: str = 'cuda',
                 unet_model_path: str = None,
                 frcnn_model_path: str = None):

        super(UNetWatershed, self).__init__()

        self._in_channels = 1
        self.device = device

        self.model = torch.jit.script(WatershedUnet(in_channels=self._in_channels)).to(device)

        checkpoint = torch.load(unet_model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.ffcnn = rcnn(path=frcnn_model_path).to(device).eval()

        self.predict_cell_candidates = functional.PredictCellCandidates(self.ffcnn, device=device)
        self.predict_segmentation_mask = functional.PredictSemanticMask(model=self.model, device=device)
        self.generate_seed_map = functional.GenerateSeedMap()

        self.instance_mask_from_prob = functional.InstanceMaskFromProb()

    def forward(self, image: torch.Tensor, channel: Optional[int] = 1) -> torch.Tensor:
        """
        Inputs an image and outputs a probability mask of everything seen in the image

        :param image: [B, C=4, X, Y, Z] torch.Tensor input image
        :param channel: int channel index of cytosol stain
        :return: [B, N, X, Y, Z] torch.Tensor cell segmenation mask wehre N is the number of cells.
        """
        assert image.ndim == 5
        if image.shape[1] != 4: raise ValueError('Need 4D image for unet+watershed segmentation.')

        image_frcnn = image[:, [0, 2, 3], :, :, :]

        cell_candidates: Dict[str, torch.Tensor] = self.predict_cell_candidates(image_frcnn)
        semantic_mask: torch.Tensor = self.predict_segmentation_mask(image[:, [channel], ...])
        seed: torch.Tensor = self.generate_seed_map(cell_candidates, semantic_mask)
        instance_mask: torch.Tensor = self.instance_mask_from_prob(semantic_mask, seed)

        return self._colormask_to_mask(instance_mask)

    @staticmethod
    def _colormask_to_mask(mask: torch.Tensor) -> torch.Tensor:
        """
        Converts a integer mask from the watershed algorithm to a 4d matrix where tensor.shape[1] is the number of
        unique cells in the mask.

        :param mask: [B, 1, X, Y, Z] int torch.Tensor where each integer is a unique cell
        :return: [B, N, X, Y, Z] float torch.Tensor where each N is a single cell
        """
        b, _, x, y, z = mask.shape
        n = len(mask.unique()) - 1  # subtract 1 because background is included
        n = n if n > 0 else 0

        out = torch.zeros((b, n, x, y, z))
        unique = mask.unique()
        unique = unique[unique != 0]

        for i, u in enumerate(unique):
            if u == 0:
                continue
            out[0, i, ...] = (mask[0, 0, ...] == u).float()

        return out
