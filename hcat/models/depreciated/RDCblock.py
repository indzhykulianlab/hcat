import torch
import torch.nn as nn
from src.models.depreciated.StackedDilation import StackedDilation
from warnings import filterwarnings

filterwarnings("ignore", category=UserWarning)


class RDCBlock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super(RDCBlock, self).__init__()

        self.conv1x1 = nn.Conv3d(in_channels * 2, in_channels, kernel_size=1)
        self.grouped_conv = StackedDilation(in_channels, in_channels, kernel=5)
        self.activation = nn.ReLU()

        self.batch_norm_conv = nn.BatchNorm3d(in_channels)
        self.batch_norm_group = nn.BatchNorm3d(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.activation(self.batch_norm_conv(self.conv1x1(x)))
        x = self.activation(self.batch_norm_group(self.grouped_conv(x)))

        return x
