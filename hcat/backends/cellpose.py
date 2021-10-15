import torch.nn as nn
import torch
from typing import Optional
from cellpose import models
from hcat.train.dataloader import colormask_to_torch_mask
from hcat.lib.utils import graceful_exit
from hcat.backends.backend import Backend
from hcat import ShapeError

# DOCUMENTED

class Cellpose(Backend):
    """
    3D Hair Cell Segmentation Backend based on the cellpose algorithm:

    Stringer, Carsen, Wang, Tim, Michaelos, Michalis, & Pachitariu, Marius. (2021).
    Cellpose: A generalist algorithm for cellular segmentation. Nature Methods, 18(1), 100-106.

    """
    def __init__(self, device: Optional[str] = 'cuda'):
        """
        Initializes a cell segmentation backbone.

        .. warning::
           Initializes the cellpose model using the cellpose external website: https://www.cellpose.org/models/

        :param device: String representing pytorch device by which to evaluate the model on.
        """
        super(Cellpose, self).__init__()
        self.model = models.Cellpose(gpu=device, model_type='cyto')
        self.channels = [0, 0]

    @graceful_exit('\x1b[1;31;40m' + 'ERROR: Cellpose Failed. Aborting...' + '\x1b[0m')  # Throws text instead of error
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Inputs an image and outputs a probability mask of everything seen in the image.

        .. note::
           Call the module as a function to execute this method (similar to torch.nn.module).

        .. warning::
           Quirks in execution means that the image will move from gpu to cpu then back.

        .. warning:
           Will not raise an error upon failure, instead returns None and prints to standard out


        Example:

        >>> from hcat.backends.cellpose import Cellpose
        >>> import torch
        >>> backend = Cellpose()
        >>> image = torch.load('path/to/my/image.trch')
        >>> assert image.ndim == 5 # Shape should be [B, C, X, Y, Z]
        >>> masks = backend(image)


        :param image: [B, C=4, X, Y, Z] input image
        :return: [B, N, X, Y, Z] output segmentation mask where each pixel value is a cell id (0 is background)
        """

        if image.ndim != 5: raise ShapeError(f'Image ndim should be 5 with shape [B, C, X, Y, Z] not {image.ndim} with shape {image.shape}')
        b, c, x, y, z = image.shape

        if self.image_reject and self._is_image_bad(image):
            return torch.zeros((b, 0, x, y, z), device=self.device)

        masks, flows, styles, diams = self.model.eval(image.cpu().numpy(), diameter=None, channels=self.channels,
                                                      channel_axis=1, z_axis=-1, do_3D=True,
                                                      anisotropy=3, progress=False)

        masks = torch.from_numpy(masks.transpose(1, 2, 0)).unsqueeze(0)

        # Converts a mask of shape [B, C=1, X, Y, Z] where each pixel is a cell id (0 for background)
        # to a mask of shape [B, N, X, Y, Z] where N is the number of cells
        masks = colormask_to_torch_mask(masks)

        return masks.unsqueeze(0)
