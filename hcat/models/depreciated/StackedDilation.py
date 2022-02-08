import torch
import torch.nn as nn
from warnings import filterwarnings
from typing import Tuple, Union

filterwarnings("ignore", category=UserWarning)

class StackedDilation(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel: Union[int, Tuple[int, int, int]],
                 dilation: Union[int, Tuple[int, int, int]] = None,
                 padding: Union[int, Tuple[int, int, int]] = None
                 ):

        super(StackedDilation, self).__init__()

        self.dilation = dilation if dilation is not None else (1, 2, 3, 4)
        self.padding = padding if padding is not None else (2, 4, 6, 8)

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel, dilation=1, padding=2)
        self.out_conv = nn.Conv3d(out_channels*len(self.dilation), out_channels, kernel_size=1, padding=0)
        self.activation = nn.ReLU()
        self.batch_norm_conv = nn.BatchNorm3d(out_channels)
        self.batch_norm_out = nn.BatchNorm3d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, (dilation, padding) in enumerate(zip(self.dilation, self.padding)):
            self.conv.dilation = dilation
            self.conv.padding = padding
            x = self.activation(self.batch_norm_conv(self.conv(x)))
            if i == 0:
                out = x
            else:
                out = torch.cat((out, x), dim=1)
        out = self.activation(self.batch_norm_out(self.out_conv(out)))
        return out
