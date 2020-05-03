import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, conv_dim=32):
        super(Discriminator, self).__init__()
        ## 128x128x3
        self.conv1 = self.conv(3, conv_dim*2) ## 64x64x64
        self.conv2 = self.conv(conv_dim*2, conv_dim*4, batch_norm=True) ## 32x32x128
        self.conv3 = self.conv(conv_dim*4, conv_dim*8, batch_norm=True) ## 16x16x256
        self.conv4 = self.conv(conv_dim*8, conv_dim*16, kernel_size=3, stride=1, batch_norm=True) ## 16x16x512
        self.conv5 = self.conv(conv_dim*16, conv_dim*32, batch_norm=True) ## 8x8x1024


        self.fc = nn.Linear(conv_dim*32*8*8, 1)

    def conv(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=False):
        layers = []
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        layers.append(conv)
        if batch_norm:
            batch_norm_layer = nn.BatchNorm2d(out_channels)
            layers.append(batch_norm_layer)

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = F.leaky_relu(self.conv5(x), 0.2)

        x = x.view(-1, 8*8*1024)
        x = self.fc(x)

        return x

class Generator(nn.Module):
    def __init__(self, z_size, conv_dim=32):
        super(Generator, self).__init__()

        self.fc = nn.Linear(z_size, 8*8*32*conv_dim)
        ## output of deconv , So=stride(Si−1)+Sf−2∗pad
        # where So means output size, Si-input size, Sf- filter size.
        ## 8x8x1024
        self.deconv1 = self.deconv(conv_dim*32, conv_dim*16, batch_norm=True) ## 16x16x512
        self.deconv2 = self.deconv(conv_dim*16, conv_dim*8, batch_norm=True) ## 32x32x256
        self.deconv3 = self.deconv(conv_dim*8, conv_dim*4, batch_norm=True) ## 64x64x128
        self.deconv4 = self.deconv(conv_dim*4, conv_dim*2, batch_norm=True) ## 128x128x64
        self.deconv5 = self.deconv(conv_dim*2, 3, kernel_size=3, stride=1) ## 128x128x3

    def deconv(self, in_channel, out_channel, kernel_size=4, stride=2, padding=1, batch_norm=False):
        layers = []
        decon = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding, bias=False)
        layers.append(decon)
        if batch_norm:
            batch_layer = nn.BatchNorm2d(out_channel)
            layers.append(batch_layer)

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc(x)

        x = x.view(-1, 1024, 8, 8)

        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        x = torch.tanh(self.deconv5(x))

        return x
