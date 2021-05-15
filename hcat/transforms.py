import torch
import torchvision.transforms.functional
import torch.nn.functional as F
from kornia.augmentation import RandomAffine3D
import numpy as np
from typing import Dict, Tuple, Union, Sequence, List
import elasticdeform
import skimage.io as io
from hcat.exceptions import ShapeError


# -------------------------------- Assumptions ------------------------------#
#                Every image is expected to be [C, X, Y, Z]                  #
#        Every transform's input has to be Dict[str, torch.Tensor]           #
#       Every transform's output has to be Dict[str, torch.Tensor]           #
#       Preserve whatever device the tensor was on when it started           #
# ---------------------------------------------------------------------------#

class transform:
    """ parent class to every transform. """
    def __init__(self):
        pass
    def check_inputs(self, x: Dict[str, torch.Tensor]) -> None:
        """
        Checks inputs of a transform and raises errors if they are bad.

        :param x: Dict[str, torch.Tensor] input to transform
        :return: None
        :raises: RuntimeError: Raises Error if shapes are not expected
        :raises: ValueError: Raises Error if key is not a string, or value is not a tensor
        :raises: KeyError: Raises Error if expected keys are not in the dictionary
        """
        if not isinstance(x, dict): raise ValueError(f'Expected input to be Dict not {type(x)}')

        for key in x:
            if not isinstance(key, str):
                raise ValueError(f'Key: {key} in input dictionary is not type str, found type: {type(key)}')

        for key in x:
            if not isinstance(x[key], torch.Tensor):
                raise ValueError(f'Value for key: {key} in input dictionary is not of type torch.Tensor, found type: {type(x[key])}')

        if 'image' not in x: raise KeyError('key "image" not found in input dictionary.')
        if 'masks' not in x: raise KeyError('key "masks" not found in input dictionary.')
        if 'centroids' not in x: raise KeyError('key "centroids" not found in input dictionary.')

        if x['image'].shape[1::] != x['masks'].shape[1::]:
            raise ShapeError(f'Shape of input["image"]: {x["image"].shape} != '
                               f'shape of input["masks"]: {x["masks"].shape}')

        for key in x:
            if x[key].device != x['image'].device:
                raise RuntimeError(f'Tensors do not share same device.\n'
                                   f'\t"image": {x["image"].device}\n'
                                   f'\t"masks": {x["masks"].device}\n'
                                   f'\t"centroids": {x["centroids"].device}\n')

        if x['image'].ndim != 4: raise ShapeError(f'Expected image.ndim = 4, not: {x["image"].ndim}')
        if x['masks'].ndim != 4: raise ShapeError(f'Expected mask.ndim = 4, not: {x["masks"].ndim} ')


class adjust_brightness(transform):
    def __init__(self, rate: float = 0.5, range_brightness: Tuple[float, float] = (-0.5, 0.5)) -> None:
        super(adjust_brightness, self).__init__()
        self.rate = rate
        self.range = range_brightness
        self.fun = self._adjust_brightness

    def __call__(self, data_dict):
        self.check_inputs(data_dict)

        if torch.rand(1) < self.rate:
            # funky looking but FAST
            val = torch.FloatTensor(data_dict['image'].shape[0]).uniform_(self.range[0], self.range[1])
            data_dict['image'] = self.fun(data_dict['image'], val)

        return data_dict

    @staticmethod
    @torch.jit.script
    def _adjust_brightness(img: torch.Tensor, val: torch.Tensor) -> torch.Tensor:
        """
        Adjusts brigtness of img with val

        :param img: [C, X, Y, Z]
        :param val: Tensor[float] [C]
        :return:
        """
        img = img.add_(val.reshape(img.shape[0], 1, 1, 1).to(img.device))
        img[img < 0] = 0
        img[img > 1] = 1
        return img


class adjust_gamma(transform):
    def __init__(self, rate: float = 0.5, gamma: Tuple[float, float] = (0.5, 1.5),
                 gain: Tuple[float, float] = (.75, 1.25)) -> None:
        super(adjust_gamma, self).__init__()
        self.rate = rate
        self.gain = gain
        self.gamma = gamma

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Randomly adjusts gamma of image color channels independently

        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
            key : val
            'image' : torch.Tensor of size [C, X, Y, Z] where C is the number of colors, X,Y,Z are the mask height,
                      width, and depth
            'masks' : torch.Tensor of size [I, X, Y, Z] where I is the number of identifiable objects in the mask
            'centroids' : torch.Tensor of size [I, 3] where dimension two is the [X, Y, Z] position of the centroid
                          for instance i

        :return: data_dict Dict[str, torch.Tensor]: dictonary with identical keys as input, but with transformed values
        """
        self.check_inputs(data_dict)

        if torch.rand(1) < self.rate:
            gamma = torch.FloatTensor(data_dict['image'].shape[0]).uniform_(self.gamma[0], self.gamma[1])
            gain = torch.FloatTensor(data_dict['image'].shape[0]).uniform_(self.gain[0], self.gain[1])

            data_dict['image'] = self._adjust_gamma(data_dict['image'], gamma, gain)

        return data_dict

    @staticmethod
    @torch.jit.script
    def _adjust_gamma(img: torch.Tensor, gamma: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
        """
        Assume img in shape [C, X, Y, Z]

        :param img:
        :param gamma:
        :param gain:
        :return:
        """
        for c in range(img.shape[0]):
            img[c, ...] = torchvision.transforms.functional.adjust_gamma(img[c, ...], gamma=gamma[c], gain=gain[c])
        return img


class affine3d(transform):
    def __init__(self,
                 rate: float = 0.75,
                 angle: Dict[str, Tuple[float, float]] = {'yaw': (-180., 180.), 'pitch': (-20., 20.), 'roll': (-20., 20.)},
                 shear: Tuple[float, float] = (-5., 5.),
                 scale: Tuple[float, float] = (0.8, 1.2)) -> None:
        """
        yaw: rotate around Y
        pitch: rotate around x
        roll: rotate around z

        :param rate:
        :param angle:
        :param shear:
        :param scale:
        """
        super(affine3d, self).__init__()
        self.rate = rate
        self.angle = angle
        self.shear = shear
        self.scale = scale

        for key in self.angle:
            if not isinstance(self.angle[key][0], float):
                raise ValueError(f'Angle must be a float: {key}, {self.angle[key][0]}')
            if not isinstance(self.angle[key][1], float):
                raise ValueError(f'Angle must be a float: {key}, {self.angle[key][1]}')

        s = (self.angle['yaw'], self.angle['pitch'], self.angle['roll'])
        self.fun = RandomAffine3D(degrees=s, shears=self.shear, scale=self.scale, resample=0, p=1)

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Performs a 3d affine transformation on the image and mask
        Shears, rotates and scales the image randomly based on parameters defined at initialization

        Adjusts for anisotropy stupidly

        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
            key : val
            'image' : torch.Tensor of size [C, X, Y, Z] where C is the number of colors, X,Y,Z are the mask height,
                      width, and depth
            'masks' : torch.Tensor of size [I, X, Y, Z] where I is the number of identifiable objects in the mask
            'centroids' : torch.Tensor of size [I, 3] where dimension two is the [X, Y, Z] position of the centroid
                          for instance i

        :return: data_dict Dict[str, torch.Tensor]: dictonary with identical keys as input, but with transformed values
        """
        self.check_inputs(data_dict)

        if torch.rand(1) < self.rate:
            stack = torch.cat((data_dict['image'], data_dict['masks']), dim=0).unsqueeze(0)
            b, c, x, y, z = stack.shape
            bigstack = torch.zeros((b, c, x, y, z * 3))

            # stacks are nonisotropic. Hacky way to adjust for this.
            for i in range(z):
                bigstack[..., (3 * i) + 0] = stack[..., i]
                bigstack[..., (3 * i) + 1] = stack[..., i]
                bigstack[..., (3 * i) + 2] = stack[..., i]

            bigstack = self.fun(bigstack.squeeze(0))                # Perform affine on expanded stack

            for i in range(z):
                stack[..., i] = bigstack[..., (3 * i) + 1]          # Return to nonisotropic

            data_dict['image'] = stack[0, 0:-1:1, ...]
            data_dict['masks'] = stack[0, [-1], ...]

        return data_dict


class colormask_to_mask(transform):
    def __init__(self):
        super(colormask_to_mask, self).__init__()

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Some Geometric transforms may alter the locations of cells so drastically that the centroid may no longer
        be accurate. This recalculates the centroids based on the current mask.

        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
            key : val
            'image' : torch.Tensor of size [C, X, Y, Z] where C is the number of colors, X,Y,Z are the mask height,
                      width, and depth
            'masks' : torch.Tensor of size [I, X, Y, Z] where I is the number of identifiable objects in the mask
            'centroids' : torch.Tensor of size [I, 3] where dimension two is the [X, Y, Z] position of the centroid
                          for instance i

        :return: data_dict Dict[str, torch.Tensor]: dictonary with identical keys as input, but with transformed values
        """
        self.check_inputs(data_dict)
        data_dict['masks'] = self._colormask_to_torch_mask(data_dict['masks'])
        return data_dict

    @staticmethod
    @torch.jit.script
    def _colormask_to_torch_mask(colormask: torch.Tensor) -> torch.Tensor:
        """

        :param colormask: [C=1, X, Y, Z]
        :return:
        """
        uni = torch.unique(colormask)
        uni = uni[uni != 0]
        num_cells = len(uni)

        shape = (num_cells, colormask.shape[1], colormask.shape[2], colormask.shape[3])
        mask = torch.zeros(shape, device=colormask.device)

        for i, u in enumerate(uni):
            mask[i, :, :, :] = (colormask[0, :, :, :] == u)

        return mask


class debug:
    def __init__(self, ind: int = 0):
        self.ind = ind

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        image = data_dict['image']
        mask = data_dict['masks']
        try:
            assert image.shape[-1] == mask.shape[-1]
            assert image.shape[-2] == mask.shape[-2]
            assert image.shape[-3] == mask.shape[-3]

            assert image.max() <= 1
            assert mask.max() <= 1
            assert image.min() >= 0
            assert mask.min() >= 0
        except Exception as ex:
            print(self.ind)
            raise ex

        return data_dict


class elastic_deformation(transform):
    def __init__(self, grid_shape: Tuple[int, int, int] = (2, 2, 2), scale: int = 2, rate: float = 0.5):
        super(elastic_deformation, self).__init__()
        self.x_grid = grid_shape[0]
        self.y_grid = grid_shape[1]
        self.z_grid = grid_shape[2] if len(grid_shape) > 2 else None
        self.scale = scale
        self.rate = rate

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self.check_inputs(data_dict)

        device = data_dict['image'].device
        image = data_dict['image'].cpu().numpy()
        mask = data_dict['masks'].cpu().numpy()
        dtype = image.dtype

        if torch.rand(1) < self.rate:
            displacement = np.random.randn(3, self.x_grid, self.y_grid, self.z_grid) * self.scale
            image = elasticdeform.deform_grid(image, displacement, axis=(1, 2, 3))
            mask = elasticdeform.deform_grid(mask, displacement, axis=(1, 2, 3), order=0)

            image[image < 0] = 0.0
            image[image > 1] = 1.0
            image.astype(dtype)

            data_dict['image'] = torch.from_numpy(image).to(device)
            data_dict['masks'] = torch.from_numpy(mask).to(device)

        return data_dict


class erosion(transform):
    def __init__(self, kernel_targets: int = 3, rate: float = 0.5, device: str = 'cpu') -> None:
        super(erosion, self).__init__()
        self.device = device
        self.rate = rate
        self.kernel_targets = kernel_targets

        if kernel_targets % 2 != 1:
            raise ValueError('Expected Kernel target to be Odd')

        self.padding: Tuple[int, int, int] = self._compute_zero_padding(
            (kernel_targets, kernel_targets, kernel_targets))
        self.kernel: torch.Tensor = self._get_binary_kernel3d(kernel_targets, device)

    def __call__(self, input: Union[Dict[str, torch.Tensor], torch.Tensor]) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Randomly applies a gaussian blur

        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
            key : val
            'image' : torch.Tensor of size [C, X, Y, Z] where C is the number of colors, X,Y,Z are the mask height,
                      width, and depth
            'masks' : torch.Tensor of size [I, X, Y, Z] where I is the number of identifiable objects in the mask
            'centroids' : torch.Tensor of size [I, 3] where dimension two is the [X, Y, Z] position of the centroid
                          for instance i

        :return: data_dict Dict[str, torch.Tensor]: dictionary with identical keys as input, but with transformed values
        """
        if isinstance(input, Dict):
            self.check_inputs(input)
            return self._is_dict(input)
        elif isinstance(input, torch.Tensor):
            return self._is_tensor(input)

    def _is_dict(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if data_dict['masks'].dtype != self.kernel.dtype:
            raise ValueError(f'Expected Image dtype to be {self.kernel.dtype} not {data_dict["masks"].dtype}')
        # We expect the image to have ndim==4 but need a batch size of 1 for median filter

        device = data_dict['image'].device
        if self.kernel.device != device:
            self.kernel = self.kernel.to(device)

        if torch.rand(1) < self.rate:
            im = data_dict['masks'].unsqueeze(0)
            b, c, h, w, d = im.shape
            # map the local window to single vector
            features: torch.Tensor = F.conv3d(im.reshape(b * c, 1, h, w, d),
                                              self.kernel,
                                              padding=self.padding, stride=1)
            data_dict['masks'] = features.min(dim=1)[0]
        return data_dict


    def _is_tensor(self, input: torch.Tensor) -> torch.Tensor:
        "Assume [B, C, X, Y, Z]"
        if input.device != self.kernel.device:
            raise ValueError(f'Expected Image Device to be {self.kernel.device} not {input.device}')
        if input.dtype != self.kernel.dtype:
            raise ValueError(f'Expected Image dtype to be {self.kernel.dtype} not {input.dtype}')
        if input.ndim != 4:
            raise ValueError(f'Expected Image ndim to be 4 not {input.ndim}, with shape {input.shape}')

        b, c, h, w, d = input.unsqueeze(0).shape
        # map the local window to single vector
        features: torch.Tensor = F.conv3d(input.reshape(b * c, 1, h, w, d), self.kernel,
                                          padding=self.padding, stride=1)
        return features.min(dim=1)[0]

    @staticmethod
    def _compute_zero_padding(kernel_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        r"""Utility function that computes zero padding tuple.
        Adapted from Kornia
        """
        computed: List[int] = [(k - 1) // 2 for k in kernel_size]
        return computed[0], computed[1], computed[2]

    @staticmethod
    def _get_binary_kernel3d(window_size: int, device: torch.device) -> torch.Tensor:
        r"""Creates a symetric binary kernel to extract the patches. If the window size
        is HxWxD will create a (H*W)xHxW kernel.

        ADAPTED FROM KORNIA

        """
        window_range: int = int(window_size ** 3)
        kernel: torch.Tensor = torch.zeros((window_range, window_range, window_range), device=device)
        for i in range(window_range):
            kernel[i, i, i] += 1.0
        kernel = kernel.view(-1, 1, window_size, window_size, window_size)

        # get rid of all zero kernels
        ind = torch.nonzero(kernel.view(kernel.shape[0], -1).sum(1))
        return kernel[ind[:, 0], ...]


class gaussian_blur(transform):
    def __init__(self, kernel_targets: torch.Tensor = torch.tensor([3, 5, 7]), rate: float = 0.5) -> None:
        super(gaussian_blur, self).__init__()
        self.kernel_targets = kernel_targets
        self.rate = rate
        self.fun = torch.jit.script(torchvision.transforms.functional.gaussian_blur)

    def __call__(self, data_dict):
        """
        Randomly applies a gaussian blur

        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
            key : val
            'image' : torch.Tensor of size [C, X, Y, Z] where C is the number of colors, X,Y,Z are the mask height,
                      width, and depth
            'masks' : torch.Tensor of size [I, X, Y, Z] where I is the number of identifiable objects in the mask
            'centroids' : torch.Tensor of size [I, 3] where dimension two is the [X, Y, Z] position of the centroid
                          for instance i

        :return: data_dict Dict[str, torch.Tensor]: dictionary with identical keys as input, but with transformed values
        """
        self.check_inputs(data_dict)

        if torch.rand(1) < self.rate:
            kern = self.kernel_targets[int(torch.randint(0, len(self.kernel_targets), (1, 1)).item())].item()
            data_dict['image'] = _reshape(self.fun(_shape(data_dict['image']), [kern, kern]))
        return data_dict


class median_filter(transform):
    def __init__(self, kernel_targets: int = 3, rate: float = 0.5, device: str = 'cpu') -> None:
        super(median_filter, self).__init__()
        self.device = device
        self.rate = rate
        self.kernel_targets = kernel_targets

        if kernel_targets % 2 != 1:
            raise ValueError('Expected Kernel target to be Odd')

        self.padding: Tuple[int, int, int] = self._compute_zero_padding(
            (kernel_targets, kernel_targets, kernel_targets))
        self.kernel: torch.Tensor = self._get_binary_kernel3d(kernel_targets, device)

    def __call__(self, input: Union[Dict[str, torch.Tensor], torch.Tensor]) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Randomly applies a gaussian blur

        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
            key : val
            'image' : torch.Tensor of size [C, X, Y, Z] where C is the number of colors, X,Y,Z are the mask height,
                      width, and depth
            'masks' : torch.Tensor of size [I, X, Y, Z] where I is the number of identifiable objects in the mask
            'centroids' : torch.Tensor of size [I, 3] where dimension two is the [X, Y, Z] position of the centroid
                          for instance i

        :return: data_dict Dict[str, torch.Tensor]: dictionary with identical keys as input, but with transformed values
        """
        if isinstance(input, Dict):
            self.check_inputs(input)
            return self._is_dict(input)
        elif isinstance(input, torch.Tensor):
            return self._is_tensor(input)
        else:
            raise ValueError(f'Transformation not implemented for type: {type(input)}')

    def _is_dict(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if data_dict['image'].dtype != self.kernel.dtype:
            raise ValueError(f'Expected Image dtype to be {self.kernel.dtype} not {data_dict["image"].dtype}')

        device = data_dict['image'].device
        if self.kernel.device != device:
            self.kernel = self.kernel.to(device)

        if torch.rand(1) < self.rate:
            data_dict['image'] = data_dict['image'].unsqueeze(0)
            b, c, h, w, d = data_dict['image'].shape
            # map the local window to single vector
            features: torch.Tensor = F.conv3d(data_dict['image'].reshape(b * c, 1, h, w, d), self.kernel,
                                              padding=self.padding, stride=1)
            data_dict['image'] = torch.median(features.view(b, c, -1, h, w, d), dim=2)[0].squeeze(0)

        return data_dict

    def _is_tensor(self, input: torch.Tensor) -> torch.Tensor:
        "Assume [B, C, X, Y, Z]"
        if input.device != self.kernel.device:
            raise ValueError(f'Expected Image Device to be {self.kernel.device} not {input.device}')
        if input.dtype != self.kernel.dtype:
            raise ValueError(f'Expected Image dtype to be {self.kernel.dtype} not {input.dtype}')
        if input.ndim != 5:
            raise ValueError(f'Expected Image ndim to be 5 not {input.ndim}, with shape {input.shape}')

        b, c, h, w, d = input.shape
        # map the local window to single vector
        features: torch.Tensor = F.conv3d(input.reshape(b * c, 1, h, w, d), self.kernel,
                                          padding=self.padding, stride=1)
        return torch.median(features.view(b, c, -1, h, w, d), dim=2)[0]

    @staticmethod
    def _compute_zero_padding(kernel_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        r"""Utility function that computes zero padding tuple.
        Adapted from Kornia
        """
        computed: List[int] = [(k - 1) // 2 for k in kernel_size]
        return computed[0], computed[1], computed[2]

    @staticmethod
    def _get_binary_kernel3d(window_size: int, device: torch.device) -> torch.Tensor:
        r"""Creates a symetric binary kernel to extract the patches. If the window size
        is HxWxD will create a (H*W)xHxW kernel.

        ADAPTED FROM KORNIA

        """
        window_range: int = int(window_size ** 3)
        kernel: torch.Tensor = torch.zeros((window_range, window_range, window_range), device=device)
        for i in range(window_range):
            kernel[i, i, i] += 1.0
        kernel = kernel.view(-1, 1, window_size, window_size, window_size)

        # get rid of all zero kernels
        ind = torch.nonzero(kernel.view(kernel.shape[0], -1).sum(1))
        return kernel[ind[:, 0], ...]


class normalize(transform):
    def __init__(self, mean: Sequence[float] = [0.5], std: Sequence[float] = [0.5]) -> None:
        super(transform)
        self.mean = mean
        self.std = std
        self.fun = torch.jit.script(torchvision.transforms.functional.normalize)

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """


        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
            key : val
            'image' : torch.Tensor of size [C, X, Y, Z] where C is the number of colors, X,Y,Z are the mask height,
                      width, and depth
            'masks' : torch.Tensor of size [I, X, Y, Z] where I is the number of identifiable objects in the mask
            'centroids' : torch.Tensor of size [I, 3] where dimension two is the [X, Y, Z] position of the centroid
                          for instance i

        :return: data_dict Dict[str, torch.Tensor]: dictonary with identical keys as input, but with transformed values
        """
        self.check_inputs(data_dict)

        data_dict['image'] = self.fun(data_dict['image'], self.mean, self.std)
        return data_dict


class nul_crop(transform):
    def __init__(self, rate: float = 0.80, limit_z: int = 3) -> None:
        super(nul_crop, self).__init__()
        self.rate = rate
        self.limit_z = limit_z

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Removes blank space around cells to ensure training images has something the network can learn
        Doesnt remove Z


        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
            key : val
            'image' : torch.Tensor of size [C, X, Y, Z] where C is the number of colors, X,Y,Z are the mask height,
                      width, and depth
            'masks' : torch.Tensor of size [I, X, Y, Z] where I is the number of identifiable objects in the mask
            'centroids' : torch.Tensor of size [I, 3] where dimension two is the [X, Y, Z] position of the centroid
                          for instance i

        :return: data_dict Dict[str, torch.Tensor]: dictonary with identical keys as input, but with transformed values
        """
        self.check_inputs(data_dict)

        shape = data_dict['masks'].shape

        if torch.rand(1) < self.rate:
            ind = torch.nonzero(data_dict['masks'])  # -> [I, 4] where 4 is ndims
            x_max = ind[:, 1].max().int().item()
            y_max = ind[:, 2].max().int().item()
            z_max = ind[:, 3].max().int().item()

            x = ind[:, 1].min().int().item()
            y = ind[:, 2].min().int().item()
            z = ind[:, 3].min().int().item() - self.limit_z
            z = z if z > 0 else 0

            w = x_max - x
            h = y_max - y
            d = z_max + self.limit_z - z
            d = d if d < shape[-1] else shape[-1]

            data_dict['image'] = _crop(data_dict['image'], x=x, y=y, z=0, w=w, h=h, d=d)
            data_dict['masks'] = _crop(data_dict['masks'], x=x, y=y, z=0, w=w, h=h, d=d)

        return data_dict


class random_affine(transform):
    def __init__(self, rate: float = 0.5, angle: Tuple[int, int] = (-180, 180),
                 shear: Tuple[int, int] = (-5, 5), scale: Tuple[float, float] = (0.9, 1.1)) -> None:
        super(random_affine, self).__init__()
        self.rate = rate
        self.angle = angle
        self.shear = shear
        self.scale = scale

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Performs an affine transformation on the image and mask
        Shears, rotates and scales the image randomly based on parameters defined at initialization

        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
            key : val
            'image' : torch.Tensor of size [C, X, Y, Z] where C is the number of colors, X,Y,Z are the mask height,
                      width, and depth
            'masks' : torch.Tensor of size [I, X, Y, Z] where I is the number of identifiable objects in the mask
            'centroids' : torch.Tensor of size [I, 3] where dimension two is the [X, Y, Z] position of the centroid
                          for instance i

        :return: data_dict Dict[str, torch.Tensor]: dictonary with identical keys as input, but with transformed values
        """
        self.check_inputs(data_dict)
        if torch.rand(1) < self.rate:
            angle = torch.FloatTensor(1).uniform_(self.angle[0], self.angle[1])
            shear = torch.FloatTensor(1).uniform_(self.shear[0], self.shear[1])
            scale = torch.FloatTensor(1).uniform_(self.scale[0], self.scale[1])
            translate = torch.tensor([0, 0])

            data_dict['image'] = _reshape(self._affine(_shape(data_dict['image']), angle, translate, scale, shear))
            data_dict['masks'] = _reshape(self._affine(_shape(data_dict['masks']), angle, translate, scale, shear))

        return data_dict

    @staticmethod
    @torch.jit.script
    def _affine(img: torch.Tensor, angle: torch.Tensor, translate: torch.Tensor, scale: torch.Tensor,
                shear: torch.Tensor) -> torch.Tensor:
                                                         """
                                                         Not to be publicly accessed! Only called through src.transforms.affine

                                                         A jit scriptable wrapped version of torchvision.transforms.functional.affine
                                                         Cannot, by rule, pass a dict to a torchscript function, necessitating this function


                                                         WARNING: Performs an affine transformation on the LAST TWO DIMENSIONS
                                                         ------------------------------------------------------------------------------------------------------------
                                                         call _shape(img) prior to _affine such that this transformation is performed on the X and Y dimensions
                                                         call _reshape on the output of _affine to return to [C, X, Y, Z]

                                                         correct implementation looks like
                                                         ```python
                                                         from src.transforms import _shape, _reshape, _affine
                                                         angle = torch.tensor([0])
                                                         scale = torch.tensor([0])
                                                         shear = torch.tensor([0])
                                                         translate = torch.tensor([0])
                                                         transformed_image = _reshape(_affine(_shape(img), angle, translate, scale, shear))

                                                         ```


                                                         :param img: torch.Tensor from data_dict of shape [..., X, Y]
                                                         :param angle: torch.Tensor float in degrees
                                                         :param translate: torch.Tensor translation factor. If zero, any transformations are done around center of image
                                                         :param scale: torch.Tensor float Scale factor of affine transformation, if 1, no scaling is performed
                                                         :param shear: torch.Tensor float shear factor of affine transformation, if 0, no shearing is performed
                                                         :return: torch.Tensor
                                                         """
                                                         angle = float(angle.item())
                                                         scale = float(scale.item())
                                                         shear = [float(shear.item())]
                                                         translate_list = [int(translate[0].item()), int(translate[1].item())]
                                                         return torchvision.transforms.functional.affine(img, angle, translate_list, scale, shear)


class random_crop(transform):
    def __init__(self, shape: Tuple[int, int, int] = (256, 256, 26)) -> None:
        super(random_crop, self).__init__()
        self.w = shape[0]
        self.h = shape[1]
        self.d = shape[2]

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Randomly crops an image to a designated size. If the crop size is too big, it just takes as much as it can.

        example:
            >>> import torch
            >>> from src import random_crop
            >>>
            >>> in_image = torch.rand((300, 150, 27))
            >>> masks = torch.rand((300, 150, 27)).gt(0.5)
            >>> transform = random_crop(shape = (256, 256, 26))
            >>> in_data = {'image': in_image, 'mask': masks, 'centroids': torch.Tensor([])}
            >>> out_data: Dict[str, torch.Tensor] = transform(in_data)
            >>> assert out_data['image'].shape =
                (256, 150, 26)


        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
            key : val
            'image' : torch.Tensor of size [C, X, Y, Z] where C is the number of colors, X,Y,Z are the mask height,
                      width, and depth
            'masks' : torch.Tensor of size [I, X, Y, Z] where I is the number of identifiable objects in the mask
            'centroids' : torch.Tensor of size [I, 3] where dimension two is the [X, Y, Z] position of the centroid
                          for instance i

        :return: data_dict Dict[str, torch.Tensor]: dictonary with identical keys as input, but with transformed values

        :raises: RuntimeError | Trys to find a valid image. Throws error after 10 failed attemts.
        """
        self.check_inputs(data_dict)

        shape = data_dict['image'].shape

        x_max = shape[1] - self.w if shape[1] - self.w > 0 else 1
        y_max = shape[2] - self.h if shape[2] - self.h > 0 else 1
        z_max = shape[3] - self.d if shape[3] - self.d > 0 else 1

        x = torch.randint(x_max, (1, 1)).item()
        y = torch.randint(y_max, (1, 1)).item()
        z = torch.randint(z_max, (1, 1)).item()

        # Check if the crop doesnt contain any positive labels.
        # If it does, try generating new points
        # We want to make sure every training image has something to learn
        num_try = 0
        while _crop(data_dict['masks'], x=x, y=y, z=z, w=self.w, h=self.h, d=self.d).sum() == 0:
            num_try += 1

            x = torch.randint(x_max, (1, 1)).item()
            y = torch.randint(y_max, (1, 1)).item()
            z = torch.randint(z_max, (1, 1)).item()

            if num_try > 10:
                raise RuntimeError("Exceded maximum try's to find a valid image.")


        data_dict['image'] = _crop(data_dict['image'], x=x, y=y, z=z, w=self.w, h=self.h, d=self.d)
        data_dict['masks'] = _crop(data_dict['masks'], x=x, y=y, z=z, w=self.w, h=self.h, d=self.d)

        return data_dict


class random_h_flip(transform):
    def __init__(self, rate: float = 0.5) -> None:
        super(random_h_flip, self).__init__()
        self.rate = rate
        self.fun = torch.jit.script(torchvision.transforms.functional.hflip)

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Randomly flips the mask vertically.

        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
            key : val
            'image' : torch.Tensor of size [C, X, Y, Z] where C is the number of colors, X,Y,Z are the mask height,
                      width, and depth
            'masks' : torch.Tensor of size [I, X, Y, Z] where I is the number of identifiable objects in the mask
            'centroids' : torch.Tensor of size [I, 3] where dimension two is the [X, Y, Z] position of the centroid
                          for instance i

        :return: data_dict Dict[str, torch.Tensor]: dictonary with identical keys as input, but with transformed values
        """
        self.check_inputs(data_dict)

        if torch.rand(1) < self.rate:
            data_dict['image'] = _reshape(self.fun(_shape(data_dict['image'])))
            data_dict['masks'] = _reshape(self.fun(_shape(data_dict['masks'])))

        return data_dict


class random_v_flip(transform):
    def __init__(self, rate: float = 0.5) -> None:
        super(random_v_flip, self).__init__()
        self.rate = rate
        self.fun = torch.jit.script(torchvision.transforms.functional.vflip)

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Randomly flips the mask vertically.

        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
            key : val
            'image' : torch.Tensor of size [C, X, Y, Z] where C is the number of colors, X,Y,Z are the mask height,
                      width, and depth
            'masks' : torch.Tensor of size [I, X, Y, Z] where I is the number of identifiable objects in the mask
            'centroids' : torch.Tensor of size [I, 3] where dimension two is the [X, Y, Z] position of the centroid
                          for instance i

        :return: data_dict Dict[str, torch.Tensor]: dictonary with identical keys as input, but with transformed values
        """
        self.check_inputs(data_dict)

        if torch.rand(1) < self.rate:
            data_dict['image'] = _reshape(self.fun(_shape(data_dict['image'])))
            data_dict['masks'] = _reshape(self.fun(_shape(data_dict['masks'])))

        return data_dict


class random_noise(transform):
    def __init__(self, gamma: float = 0.1, rate: float = 0.5):
        super(random_noise, self).__init__()
        self.gamma = gamma
        self.rate = rate

    def __call__(self, data_dict: Dict[str, torch.Tensor], ) -> Dict[str, torch.Tensor]:
        """
        Adds noise to the image. Noise are values between 0 and 0.3

        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
            key : val
            'image' : torch.Tensor of size [C, X, Y, Z] where C is the number of colors, X,Y,Z are the mask height,
                      width, and depth
            'masks' : torch.Tensor of size [I, X, Y, Z] where I is the number of identifiable objects in the mask
            'centroids' : torch.Tensor of size [I, 3] where dimension two is the [X, Y, Z] position of the centroid
                          for instance i

        :return: data_dict Dict[str, torch.Tensor]: dictonary with identical keys as input, but with transformed values
        """
        self.check_inputs(data_dict)

        if torch.rand(1).item() < self.rate:
            device = data_dict['masks'].device
            noise = torch.rand(data_dict['image'].shape).to(device) * torch.tensor([self.gamma]).to(device)
            data_dict['image'] = data_dict['image'] + noise

        return data_dict


class save_image:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        image = data_dict['image'][[2, 1, 0], ...].cpu().transpose(0, -1).numpy()[10, ...]
        _, mask = torch.max(data_dict['masks'].cpu(), 0)

        print(mask.max())

        mask = mask.float().numpy().transpose((2, 0, 1))[10, ...]

        io.imsave(self.name + '_image.png', image)
        io.imsave(self.name + '_mask.png', mask)

        return data_dict


class to_cuda(transform):
    def __init__(self):
        pass

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Move every element in a dict containing torch tensor to cuda.

        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
            key : val
            'image' : torch.Tensor of size [C, X, Y, Z] where C is the number of colors, X,Y,Z are the mask height,
                      width, and depth
            'masks' : torch.Tensor of size [I, X, Y, Z] where I is the number of identifiable objects in the mask
            'centroids' : torch.Tensor of size [I, 3] where dimension two is the [X, Y, Z] position of the centroid
                          for instance i

        :return: data_dict Dict[str, torch.Tensor]: dictonary with identical keys as input, but with transformed values
        """
        self.check_inputs(data_dict)
        for key in data_dict:
            data_dict[key] = data_dict[key].cuda()
        return data_dict


# class to_tensor(transform):
#     def __init__(self):
#         super(to_tensor, self).__init__()
#         pass
#
#     def __call__(self, data_dict: Dict[str, Union[torch.Tensor, Image, np.ndarray]]) -> Dict[str, torch.Tensor]:
#         """
#         Convert a PIL image or numpy.ndarray to a torch.Tensor
#
#         :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
#             key : val
#             'image' : torch.Tensor of size [C, X, Y, Z] where C is the number of colors, X,Y,Z are the mask height,
#                       width, and depth
#             'masks' : torch.Tensor of size [I, X, Y, Z] where I is the number of identifiable objects in the mask
#             'centroids' : torch.Tensor of size [I, 3] where dimension two is the [X, Y, Z] position of the centroid
#                           for instance i
#
#         :return: data_dict Dict[str, torch.Tensor]: dictonary with identical keys as input, but with transformed values
#         """
#         if not isinstance(data_dict, dict): raise ValueError(f'Does not accept input of type: {type(data_dict)}')
#         if 'image' not in data_dict: raise KeyError('key "image" not found in input dictionary.')
#         if data_dict['image'].ndim != 4: raise ShapeError('Image is not a 4D tensor.')
#
#
#         if isinstance(data_dict['image'], np.ndarray) or isinstance(data_dict['image'], Image):
#             data_dict['image'] = torchvision.transforms.functional.to_tensor(data_dict['image'])
#         return data_dict


class transformation_correction(transform):
    def __init__(self, min_cell_volume: int = 4000):
        super(transformation_correction, self).__init__()
        self.min_cell_volume = min_cell_volume

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Some Geometric transforms may alter the locations of cells so drastically that the centroid may no longer
        be accurate. This recalculates the centroids based on the current mask.

        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
            key : val
            'image' : torch.Tensor of size [C, X, Y, Z] where C is the number of colors, X,Y,Z are the mask height,
                      width, and depth
            'masks' : torch.Tensor of size [I, X, Y, Z] where I is the number of identifiable objects in the mask
            'centroids' : torch.Tensor of size [I, 3] where dimension two is the [X, Y, Z] position of the centroid
                          for instance i

        :return: data_dict Dict[str, torch.Tensor]: dictonary with identical keys as input, but with transformed values
        """
        self.check_inputs(data_dict)
        shape = data_dict['masks'].shape
        device = data_dict['masks'].device
        centroid = torch.zeros(shape[0], 3, dtype=torch.float, device=device)
        ind = torch.ones(shape[0], dtype=torch.long, device=device)

        for i in range(shape[0]):  # num of instances

            # Selectively remove mask if it is too small
            data_dict['masks'][i, ...] = self._remove_bad_masks(data_dict['masks'][i, ...], self.min_cell_volume)
            indexes = torch.nonzero(data_dict['masks'][i, ...] > 0).float()

            # If every value in a tensor is 0, assign [-1, -1, -1] as the centroid. Will be removed later
            if indexes.shape[0] == 0:
                centroid[i, :] = torch.tensor([-1, -1, -1], device=device)
                ind[i] = 0

            # If there is a mask, figure out its best location
            else:
                z_max = indexes[..., -1].max()
                z_min = indexes[..., -1].min()
                z = torch.round((z_max - z_min) / 2 + z_min) - 2
                indexes = indexes[indexes[..., -1] == z, :]
                centroid[i, :] = torch.cat((torch.mean(indexes, dim=0)[0:2], z.unsqueeze(0))).float()

            if torch.any(torch.isnan(centroid[i, :])):  # Sometimes shit is NaN, if it is, get rid of it
                centroid[i, :] = torch.tensor([-1, -1, -1], device=device)
                ind[i] == 0

        data_dict['centroids'] = centroid[ind.bool()]
        data_dict['masks'] = data_dict['masks'][ind.bool(), :, :, :]

        assert torch.isnan(data_dict['centroids']).sum() == 0

        return data_dict

    @staticmethod
    @torch.jit.script
    def _remove_bad_masks(image: torch.Tensor, min_cell_volume: int) -> torch.Tensor:
        """
        Input an boolean image volume of shape [X, Y, Z] and determine the number of nonzero pixels.

        Sets the segmentation mask of a cell to ZERO if both conditions are met:
            1 - The cell is touching an edge (All edges of volume are considered)
            2 - The cell is smaller than a lower bound set by arg: min_cell_volume

        :param image: [X, Y, Z]
        :param min_cell_volume: min number of voxels needed to keep a cell mask as valid

        :return: torch.Tensor of shape [X, Y, Z]. Returns image if no cells are touching the edge.
        """
        ind = torch.nonzero(image)  # Determine location of nonzero values -> torch.tensor([:, [X, Y, Z]])

        # Loop over each edge (i) and check if the mask is touching.
        # Implicitly, if a nonzero value is zero it is on an edge.
        for i in range(3):
            remove_bool = torch.any(ind[:, i] == 0) or torch.any(ind[:, i] == image.shape[i] - 1)
            remove_bool = remove_bool if torch.sum(image) < min_cell_volume else False

            # Remove cell if it touches the edge and is small. No need for further computation.
            if remove_bool:
                image = torch.zeros(image.shape)
                break

        return image



@torch.jit.script
def _shape(img: torch.Tensor) -> torch.Tensor:
    """
    Shapes a 4D input tensor from shape [C, X, Y, Z] to [C, Z, X, Y]

    * some torchvision functional transforms only work on last two dimensions *

    :param img: torch.Tensor image of shape [C, X, Y, Z]
    :return:
    """
    # [C, X, Y, Z] -> [C, 1, X, Y, Z] ->  [C, Z, X, Y, 1] -> [C, Z, X, Y]
    return img.unsqueeze(1).transpose(1, -1).squeeze(-1)


@torch.jit.script
def _reshape(img: torch.Tensor) -> torch.Tensor:
    """
    Reshapes a 4D input tensor from shape [C, Z, X, Y] to [C, X, Y, Z]

    Performs corrective version of _shape

    * some torchvision functional transforms only work on last two dimensions *

    :param img: torch.Tensor image of shape [C, Z, X, Y]
    :return:
    """
    # [C, Z, X, Y] -> [C, Z, X, Y, 1] ->  [C, 1, X, Y, Z] -> [C, Z, X, Y]
    return img.unsqueeze(-1).transpose(1, -1).squeeze(1)


@torch.jit.script
def _crop(img: torch.Tensor, x: int, y: int, z: int, w: int, h: int, d: int) -> torch.Tensor:
    """
    torch scriptable function which crops an image

    :param img: torch.Tensor image of shape [C, X, Y, Z]
    :param x: x coord of crop box
    :param y: y coord of crop box
    :param z: z coord of crop box
    :param w: width of crop box
    :param h: height of crop box
    :param d: depth of crop box
    :return:
    """
    if img.ndim == 4:
        img = img[:, x:x + w, y:y + h, z:z + d]
    elif img.ndim == 5:
        img = img[:, :, x:x + w, y:y + h, z:z + d]
    else:
        raise IndexError('Unsupported number of dimensions')

    return img


if __name__ == '__main__':
    pass
