import torch.nn as nn
import torch
from src.train.transforms import _crop
from typing import List


class r_unet_LTS(nn.Module):
    """
    LTS r_unet. Assumes 512 scale param

    """
    def __init__(self, in_channels=1, out_channels=4):
        super(r_unet_LTS, self).__init__()

        self.activation = nn.LeakyReLU()

        # c = [20, 25, 30]
        c = [10, 20, 30]

        self.down_1 = r_block(in_channels, c[0], 5, 1, 1)
        self.down_2 = r_block(c[0], c[1], 5, 2, 2)
        self.down_3 = r_block(c[1], c[2], 5, 3, 3)

        self.up_1 = r_block(c[2] + c[1], c[1], 5)
        self.up_2 = r_block(c[1] + c[0], c[0] + 10, 5)

        self.out_embed = nn.Conv3d(c[0] + 10, 3, kernel_size=3, stride=1, dilation=1, padding=1)
        self.out_prob = nn.Conv3d(c[0] + 10, 1, kernel_size=3, stride=1, dilation=1, padding=1)

        self.stride_1 = nn.Conv3d(c[0], c[0], kernel_size=3, stride=(3, 3, 2), dilation=1, padding=0)
        self.stride_2 = nn.Conv3d(c[1], c[1], kernel_size=3, stride=(3, 3, 2), dilation=1, padding=0)

        self.transpose_1 = nn.ConvTranspose3d(c[2], c[2], kernel_size=3, stride=(3, 3, 2), padding=0, dilation=1)
        self.transpose_2 = nn.ConvTranspose3d(c[1], c[1], kernel_size=3, stride=(3, 3, 2), padding=0, dilation=1)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.down_1(x)
        y = self.down_2(self.activation(self.stride_1(x)))
        z = self.down_3(self.activation(self.stride_2(y)))
        z = self.activation(self.transpose_1(z))
        z = self.up_1(torch.cat((crop(y, z.shape), z), dim=1))
        z = self.activation(self.transpose_2(z))
        z = self.up_2(torch.cat((crop(x, z.shape), z), dim=1))

        z = torch.cat((
            self.tanh(self.out_embed(z)),
            self.sigmoid(self.out_prob(z))
        ), dim=1)

        return z


class r_block(nn.Module):
    """

    Recurrently applies a block to an input and outputs later

    """
    def __init__(self, in_channels: int, out_channels: int, n_loop: int, dilation: int = 1, padding: int = 1):
        super(r_block, self).__init__()

        self.activation = nn.LeakyReLU()
        self.n = n_loop

        self.conv1x1 = nn.Conv3d(in_channels * 2, in_channels, kernel_size=1)
        # self.conv7x7_3 = nn.Conv3d(in_channels, in_channels, kernel_size=(3,3,1), padding=(1,1,0), stride=1, dilation=1)
        self.conv7x7_1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=padding, stride=1, dilation=dilation)
        self.conv7x7_2 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=padding, stride=1, dilation=dilation)

        self.activation = nn.LeakyReLU()

        self.batch_norm_conv_1x1 = nn.BatchNorm3d(in_channels)
        self.batch_norm_conv_7x7_1 = nn.BatchNorm3d(in_channels)
        # self.batch_norm_conv_7x7_3 = nn.BatchNorm3d(in_channels)
        self.batch_norm_conv_7x7_2 = nn.BatchNorm3d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.zeros(x.shape, device=x.device)
        for _ in range(self.n):
            y = torch.cat((x, y), dim=1)
            y = self.activation(self.batch_norm_conv_1x1(self.conv1x1(y)))
            # y = self.activation(self.batch_norm_conv_7x7_3(self.conv7x7_3(y)))
            y = self.activation(self.batch_norm_conv_7x7_1(self.conv7x7_1(y)))

        y = self.activation(self.batch_norm_conv_7x7_2(self.conv7x7_2(y)))

        return y


@torch.jit.script
def crop(x: torch.Tensor, shape: List[int]) -> torch.Tensor:
    return _crop(x, 0, 0, 0, int(shape[2]), int(shape[3]), int(shape[4]))
