import torch
import torch.nn as nn



class RecurrentUnet(nn.Module):

    def __init__(self, in_channels=4, out_channels=3):
        super(RecurrentUnet, self).__init__()
        channels = [8,16,32,64,128]
        channels = [4,8,16,32,64]

        # self.conv9 = nn.Conv3d(in_channels, channels[0], kernel_size=9, padding=4)
        self.conv7 = nn.Conv3d(in_channels, channels[1], kernel_size=7, padding=3)
        self.conv5 = nn.Conv3d(channels[1], channels[2], kernel_size=5, padding=2)
        self.conv3 = nn.Conv3d(channels[2], channels[3], kernel_size=3, padding=1)

        self.intermediate_conv = nn.Conv3d(channels[3]*2, channels[3], kernel_size=1,padding=0)

        self.fz = nn.Conv3d(channels[3], channels[4], kernel_size=3, padding=1)
        self.fh = nn.Conv3d(channels[3], channels[4], kernel_size=3, padding=1)

        self.up_conv = nn.ConvTranspose3d(channels[3], channels[3], kernel_size=3, stride=2)
        self.out_conv = nn.Conv3d(channels[3], out_channels, 1)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.max_pool = nn.MaxPool3d((2,2,2))


    def forward(self, image):
        # image = self.tanh(self.conv9(image))
        image = self.tanh(self.conv7(image))
        image = self.tanh(self.conv5(image))
        image = self.tanh(self.conv3(image))
        image = self.max_pool(image)

        for t in range(10):
            if t == 0:
                s_t = torch.zeros(image.shape).cuda()

            image = torch.cat((image, s_t), dim=1)
            image = self.intermediate_conv(image)

            # Recurrent Bit!!!
            h = self.tanh(self.fh(image))
            if t == 0:
                h_t = torch.ones(h.shape).cuda()

            z = self.sigmoid(self.fz(image))
            h_t = (h_t * z) + (-1 * z * h)

            s_t = image

        image = self.up_conv(image)


        return self.out_conv(image)
