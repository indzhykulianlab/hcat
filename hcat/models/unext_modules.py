import torch
from torch import Tensor
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union
import torch.nn as nn
import math

# @torch.jit.script
def drop_path(x: Tensor, drop_prob: float = 0., training: bool = False) -> Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    TIMM

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x

    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim: int, drop_path: Optional[float] = 0.0, kernel_size: int = 7, dilation: int = 1,
                 layer_scale_init_value: Optional[float] = 1e-2, activation: Optional = None) -> None:
        super().__init__()

        padding = kernel_size // 2
        kernel_size = (kernel_size, ) * 3 if isinstance(kernel_size, int) else kernel_size

        self.dwconv = nn.Conv3d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim, dilation=dilation)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = activation() if activation else nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path: Union[DropPath, nn.Identity] = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.dim = dim

    def __repr__(self):
        out = f'Block[In={self.dim}, Out={self.dim}]' \
              f'\n\tDepthWiseConv[In={self.dwconv.in_channels}, Out={self.dwconv.out_channels}, KernelSize={self.dwconv.kernel_size}]' \
              f'\n\tLayerNorm[Shape={self.norm.normalized_shape}]' \
              f'\n\tPointWiseConv[In={self.pwconv1.in_features}, Out={self.pwconv1.out_features}]' \
              f'\n\tGELU[]' \
              f'\n\tPointWiseConv[In={self.pwconv2.in_features}, Out={self.pwconv2.out_features}]' \
              f'\n\tScale[]' \
              f'\n\tDropPath[]'

        return out

    def forward(self, x: Tensor) -> Tensor:  # Image of shape (B, C, H, W) -> (10, 3, 100, 100)
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, num_features, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (num_features,)

        # small optimization. Avoids repeated branching in forward call...
        self.norm_functon = F.layer_norm if self.data_format == 'channels_last' else self.layer_norm_channels_fist

    def forward(self, x: Tensor) -> Tensor:
        # if self.data_format == "channels_last":
        #     x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # elif self.data_format == "channels_first":
        #     # u = x.mean(1, keepdim=True)
        #     # s = (x - u).pow(2).mean(1, keepdim=True)
        #     # x = (x - u) / torch.sqrt(s + self.eps)
        #     # x = self.weight.view(1, -1, 1, 1, 1) * x + self.bias.view(1, -1, 1, 1, 1)
        #     x = self.layer_norm_channels_fist(x, self.weight, self.bias, self.eps)

        return self.norm_functon(x, self.normalized_shape, self.weight, self.bias, self.eps)

    @staticmethod
    # @torch.jit.script  # Fuze operation to potentially take place on a single cuda kernel
    def layer_norm_channels_fist(x: Tensor, shape: Tuple[int], weight: Tensor, bias: Tensor, eps: float) -> Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + eps)
        return weight.view(1, -1, 1, 1, 1) * x + bias.view(1, -1, 1, 1, 1)


class UpSampleLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, method='nearest'):
        super(UpSampleLayer, self).__init__()
        self.method = method
        self.norm = LayerNorm(in_channels, eps=1e-6, data_format="channels_first")
        self.compress = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=1)

    def forward(self, x: Tensor, shape: List[int]) -> Tensor:
        'Upsample layer'
        # print(x.shape ,x.numel(), x.max(), x.min())
        x = F.interpolate(x, shape[2::], mode=self.method)
        x = self.norm(x)
        x = self.compress(x)
        return x


class ConcatConv(nn.Module):
    def __init__(self, in_channels: int):
        super(ConcatConv, self).__init__()
        self.conv = nn.Conv3d(in_channels * 2, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # print(f'ConcatConv Called: x: {x.shape}, y: {y.shape}')
        x = torch.cat((x, y), dim=1)
        x = self.conv(x)
        return x


class r_block(nn.Module):
    """
    Recurrently applies a block to an input and outputs later.
    """

    def __init__(self, in_channels: int, out_channels: int, n_loop: int, dilation: Tuple[int] = (1, 1, 1),
                 padding: int = 3):
        super(r_block, self).__init__()

        self.n = n_loop

        self.conv1x1 = nn.Conv3d(in_channels * 2, in_channels, kernel_size=(1, 1, 1))
        self.conv7x7_1 = nn.Conv3d(in_channels, in_channels, kernel_size=(7, 7, 7), padding=padding, stride=(1, 1, 1),
                                   dilation=dilation)
        self.conv7x7_2 = nn.Conv3d(in_channels, out_channels, kernel_size=(7, 7, 7), padding=padding, stride=(1, 1, 1),
                                   dilation=dilation)

        self.activation = nn.LeakyReLU()

        self.batch_norm_conv_1x1 = nn.BatchNorm3d(in_channels)
        self.batch_norm_conv_7x7_1 = nn.BatchNorm3d(in_channels)
        self.batch_norm_conv_7x7_2 = nn.BatchNorm3d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        y = torch.zeros(x.shape, device=x.device)
        for _ in range(self.n):
            y = torch.cat((x, y), dim=1)
            y = self.activation(self.batch_norm_conv_1x1(self.conv1x1(y)))
            # y = self.activation(self.batch_norm_conv_7x7_3(self.conv7x7_3(y)))
            y = self.activation(self.batch_norm_conv_7x7_1(self.conv7x7_1(y)))

        y = self.activation(self.batch_norm_conv_7x7_2(self.conv7x7_2(y)))

        return y


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
