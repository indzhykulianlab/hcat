import torch
import torch.nn as nn
from warnings import filterwarnings

filterwarnings("ignore", category=UserWarning)


@torch.jit.script
def crop(x, y):
    """
    Cropping Function to crop tensors to each other. By default only crops last 2 (in 2d) or 3 (in 3d) dimensions of
    a tensor.
    :param x: Tensor to be cropped
    :param y: Tensor by who's dimmension will crop x
    :return:
    """
    shape_x = x.shape
    shape_y = y.shape
    cropped_tensor = torch.empty(0)

    assert shape_x[1] == shape_y[1],\
        f'Inputs do not have same number of feature dimmensions: {shape_x} | {shape_y}'

    if len(shape_x) == 4:
        cropped_tensor = x[:, :, 0:shape_y[2]:1, 0:shape_y[3]:1]
    if len(shape_x) == 5:
        cropped_tensor = x[:, :, 0:shape_y[2]:1, 0:shape_y[3]:1, 0:shape_y[4]:1]

    return cropped_tensor


class Up(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel: tuple,
                 upsample_kernel: tuple,
                 upsample_stride: int,
                 dilation: dict,
                 groups: dict,
                 padding_down=None,
                 padding_up=None
                 ):

        super(Up, self).__init__()

        padding_down = padding_down if padding_down is not None else 0
        padding_up = padding_up if padding_up is not None else 0

        self.conv1 = nn.Conv3d(in_channels,
                               out_channels,
                               kernel['conv1'],
                               dilation=dilation['conv1'],
                               groups=groups['conv1'],
                               padding=padding_down)

        self.conv2 = nn.Conv3d(out_channels,
                               out_channels,
                               kernel['conv2'],
                               dilation=dilation['conv2'],
                               groups=groups['conv2'],
                               padding=padding_down)

        self.up_conv = nn.ConvTranspose3d(in_channels,
                                          out_channels,
                                          upsample_kernel,
                                          stride=upsample_stride,
                                          padding=padding_up)
        self.lin_up = False

        self.batch1 = nn.BatchNorm3d(out_channels)
        self.batch2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        x = self.up_conv(x)
        y = crop(x, y)
        x = torch.cat((x, y), dim=1)
        x = self.relu(self.batch1(self.conv1(x)))
        x = self.relu(self.batch2(self.conv2(x)))
        return x
