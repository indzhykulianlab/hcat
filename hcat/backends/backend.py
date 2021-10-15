import torch.nn as nn
import wget
import os.path
import torch
import hcat
import re

class Backend(nn.Module):
    def __init__(self):
        super(Backend, self).__init__()

        self.image_reject=True

    def no_reject(self):
        """
        Disables image rejection; forces backend model to FULLY evaluate every image.

        Image rejection is on by default, disabling may significantly degrade overall performance.

        Example:

        >>> from hcat.backends.spatial_embedding import SpatialEmbedding
        >>> backend = SpatialEmbedding()
        >>>
        >>> url = 'https://www.model_location.com/model.trch'
        >>> backend.load(url) # Works with url
        >>>
        >>> model_path = 'path/to/my/model.trch'
        >>> backend.load(model_path) # Also works with path


        """
        self.image_reject = False

    def reject(self):
        """
        Enables image rejection criteria. May result in significant whole cochlea speedup.

        Image rejection is on by default, significantly improving whole image analysis speed.
        """
        self.image_reject = True

    @staticmethod
    @torch.jit.script
    def _is_image_bad(image: torch.Tensor, min_threshold: float = 0.05):
        """
        Check if an image is likely to NOT contain any cells.
        Uses cytosolic stain threshold.

        :param image: input torch tensor
        :param min_threshold: minimum value as percentage of saturated voxels
        :return:
        """
        is_bad = False

        brightness_threshold = torch.tensor(3500).div(2 ** 16).sub(0.5).div(0.5)

        if image.max() == -1:
            is_bad = True
        elif torch.sum(image.gt(brightness_threshold)) < (image.numel() * min_threshold):
            is_bad = True

        return is_bad

    @staticmethod
    def _model_loader_path(path: str, model, device: str):
        """ Loads model from a path """
        try:
            model = torch.jit.script(model).to(device)
            if path is not None:
                checkpoint = torch.load(path)
                if isinstance(checkpoint, dict):
                    checkpoint = checkpoint['model_state_dict']
                model.load_state_dict(checkpoint)
                print('src.lib.backend._model_loader_path: model successfully loaded.')

        except RuntimeError:  # This is likely due to model weights not lining up.
            model = torch.jit.script(model(1, 4).requires_grad_(False)).to(device)
            if path is not None:
                checkpoint = torch.load(path)
                if isinstance(checkpoint, dict):
                    checkpoint = checkpoint['model_state_dict']
                model.load_state_dict(checkpoint)

        for m in model.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.eval()

        return model

    @staticmethod
    def _model_loader_url(url: str, model, device: str):
        """ loads model from url """
        path = os.path.join(hcat.__path__[0], 'spatial_embedding.trch')


        if not os.path.exists(path):
            print('Downloading Model File: ')
            wget.download(url=url, out=path)
            print(' ')

        model = torch.jit.script(model(in_channels=1).requires_grad_(False))
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model = model.to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

        for m in model.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.eval()

        return model

    def _is_url(self, input: str):
        regex = re.compile(
            r'^(?:http|ftp)s?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)

        # Return true if its a url
        return re.match(regex, input) is not None


    @staticmethod
    def _colormask_to_mask(mask: torch.Tensor) -> torch.Tensor:
        """
        Converts a integer mask from the watershed algorithm to a 4d matrix where tensor.shape[1] is the number of
        unique cells in the mask.

        :param mask: [B, 1, X, Y, Z]  where each integer is a unique cell
        :return: [B, N, X, Y, Z] where each N is a single cell
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

