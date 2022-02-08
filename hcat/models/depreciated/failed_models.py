import torch
import torch.nn as nn
from src.modules import RDCBlock, Up, Down, f
from warnings import filterwarnings

filterwarnings("ignore", category=UserWarning)

class RDCNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RDCNet, self).__init__()

        complexity = 10

        self.strided_conv = nn.Conv3d(in_channels, complexity, kernel_size=3, stride=2, padding=1)
        self.RDCblock = RDCBlock(complexity)
        self.out_conv = nn.Conv3d(complexity, out_channels=complexity, kernel_size=3,padding=1)
        self.transposed_conv = nn.ConvTranspose3d(in_channels=complexity, out_channels=out_channels,
                                                  stride=(2,2,2), kernel_size=(4, 4, 4), padding=(1,1,1))

    def forward(self, x):
        x = self.strided_conv(x)
        for t in range(10):
            if t == 0:
                y = torch.zeros(x.shape).cuda()
            in_ = torch.cat((x, y), dim=1)
            y = self.RDCblock(in_) + y
        y = self.out_conv(y)
        return self.transposed_conv(y)


class RecurrentUnet_fail(nn.Module):

    def __init__(self, in_channels=4, out_channels=3):
        super(RecurrentUnet_fail, self).__init__()
        channels = [8,16,32,64,128, 256]
        channels = [5,6,7,8,16,32]

        self.conv9 = nn.Conv3d(in_channels, channels[0], kernel_size=9, padding=4)
        self.conv7 = nn.Conv3d(channels[0], channels[1], kernel_size=7, padding=3)
        self.conv5 = nn.Conv3d(channels[1], channels[2], kernel_size=5, padding=2)
        self.conv3 = nn.Conv3d(channels[2], channels[3], kernel_size=3, padding=1)

        self.intermediate_conv = nn.Conv3d(channels[3]*2, channels[3], kernel_size=1,padding=0)

        kernel = {'conv1': (3, 3, 3), 'conv2': (3, 3, 3)}
        upsample_kernel = (6, 6, 5)
        max_pool_kernel = (2, 2, 1)
        upsample_stride = (2, 2, 1)
        dilation = {'conv1': 1, 'conv2': 1}
        groups = {'conv1': 1, 'conv2': 1}

        # f_z
        self.down2_fz = Down(in_channels=channels[3], out_channels=channels[4],
                             kernel=kernel, dilation=dilation, groups=groups,padding=1)
        self.down3_fz = Down(in_channels=channels[4], out_channels=channels[5],
                             kernel=kernel, dilation=dilation, groups=groups,padding=1)
        self.up1_fz = Up(in_channels=channels[5], out_channels=channels[4], kernel=kernel, dilation=dilation, groups=groups,
                         upsample_kernel=upsample_kernel, upsample_stride = upsample_stride, padding_down=1,padding_up=2)

        # f_h
        self.down2_fh = Down(in_channels=channels[3], out_channels=channels[4],
                             kernel=kernel, dilation=dilation, groups=groups, padding=1)
        self.down3_fh = Down(in_channels=channels[4], out_channels=channels[5],
                             kernel=kernel, dilation=dilation, groups=groups, padding=1)
        self.up1_fh = Up(in_channels=channels[5], out_channels=channels[4], kernel=kernel, dilation=dilation, groups=groups,
                         upsample_kernel=upsample_kernel, upsample_stride=upsample_stride, padding_down=1, padding_up=2)

        self.out_conv = nn.Conv3d(channels[4], out_channels, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.max_pool = nn.MaxPool3d(max_pool_kernel)

        self.fz = f(self.down2_fz, self.down3_fz, self.up1_fz, self.max_pool)
        self.fh = f(self.down2_fh, self.down3_fh, self.up1_fh, self.max_pool)

    def forward(self, image):
        image = self.conv9(image)
        image = self.conv7(image)
        image = self.conv5(image)
        image = self.conv3(image)
        image = self.max_pool(image)

        for t in range(10):
            if t == 0:
                s_t = torch.zeros(image.shape).cuda()

            x = torch.cat((image, s_t), dim=1)
            x = self.intermediate_conv(x)

            # Recurrent Bit!!!
            h = self.tanh(self.fh(x))
            if t == 0:
                h_t = torch.ones(h.shape).cuda()

            z = self.sigmoid(self.fz(x))
            h_t = (h_t * z) + (-1 * z * h)

            s_t = x

        return self.out_conv(x)