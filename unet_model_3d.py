# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F

from unet_parts_3d import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, inner):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, inner)
        self.down1 = down(inner, inner)
        self.down2 = down(inner, inner)
        self.down3 = down(inner, inner)
        self.down4 = down(inner, inner)
        self.up1 = up(2 * inner, inner)
        self.up2 = up(2 * inner, inner)
        self.up3 = up(2 * inner, inner)
        self.up4 = up(2 * inner, inner)
        self.outc = outconv(inner, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down2(x3)
        x5 = self.down2(x4)
        x = self.up1(x5, x4)
        x = self.up1(x, x3)
        x = self.up1(x, x2)
        x = self.up2(x, x1)
        x = self.outc(x)
        return x
