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


class Down(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel: dict,
                 dilation: dict,
                 groups: dict,
                 padding=None
                 ):
        super(Down, self).__init__()
        if padding is None:
            padding = 0

        self.conv1 = nn.Conv3d(in_channels,
                                       out_channels,
                                       kernel['conv1'],
                                       dilation=dilation['conv1'],
                                       groups=groups['conv1'],
                                       padding=padding)

        self.conv2 = nn.Conv3d(out_channels,
                                       out_channels,
                                       kernel['conv2'],
                                       dilation=dilation['conv2'],
                                       groups=groups['conv2'],
                                       padding=1)

        self.batch1 = nn.BatchNorm3d(out_channels)
        self.batch2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.batch1(self.conv1(x)))
        x = self.relu(self.batch2(self.conv2(x)))
        return x
