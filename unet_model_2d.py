# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F

from unet_parts_2d import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, start_size=64):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, start_size)
        self.down1 = down(start_size, 2 * start_size)
        self.down2 = down(2 * start_size, 4 * start_size)
        self.down3 = down(4 * start_size, 8 * start_size)
        self.down4 = down(8 * start_size, 8 * start_size)
        self.up1 = up(16 * start_size,  4 * start_size)
        self.up2 = up(8 * start_size, 2 * start_size)
        self.up3 = up(4 * start_size, start_size)
        self.up4 = up(2 * start_size, start_size)
        self.outc = outconv(start_size, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
