"""Pix2Pix models: U-Net generator + 70x70 PatchGAN discriminator (256x256 input)."""
import torch
import torch.nn as nn


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, norm=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)]
        if norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=False):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
                  nn.BatchNorm2d(out_ch)]
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x, skip):
        return torch.cat([self.block(x), skip], dim=1)  # skip connection


class UNetGenerator(nn.Module):
    """Encoder-decoder with skip connections. Input/Output: 3x256x256 in [-1, 1]."""

    def __init__(self, in_ch=3, out_ch=3):
        super().__init__()
        self.d1 = Down(in_ch, 64, norm=False)  # 128
        self.d2 = Down(64, 128)                # 64
        self.d3 = Down(128, 256)               # 32
        self.d4 = Down(256, 512)               # 16
        self.d5 = Down(512, 512)               # 8
        self.d6 = Down(512, 512)               # 4
        self.d7 = Down(512, 512)               # 2
        self.d8 = Down(512, 512, norm=False)   # 1 (bottleneck)

        self.u1 = Up(512, 512, dropout=True)
        self.u2 = Up(1024, 512, dropout=True)
        self.u3 = Up(1024, 512, dropout=True)
        self.u4 = Up(1024, 512)
        self.u5 = Up(1024, 256)
        self.u6 = Up(512, 128)
        self.u7 = Up(256, 64)
        self.final = nn.Sequential(nn.ConvTranspose2d(128, out_ch, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        d1 = self.d1(x); d2 = self.d2(d1); d3 = self.d3(d2); d4 = self.d4(d3)
        d5 = self.d5(d4); d6 = self.d6(d5); d7 = self.d7(d6); d8 = self.d8(d7)
        x = self.u1(d8, d7); x = self.u2(x, d6); x = self.u3(x, d5)
        x = self.u4(x, d4); x = self.u5(x, d3); x = self.u6(x, d2)
        x = self.u7(x, d1)
        return self.final(x)


class PatchDiscriminator(nn.Module):
    """Judges overlapping 70x70 patches of (sketch, image) pairs as real/fake."""

    def __init__(self, in_ch=6):
        super().__init__()
        self.net = nn.Sequential(
            Down(in_ch, 64, norm=False),
            Down(64, 128),
            Down(128, 256),
            nn.Conv2d(256, 512, 4, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 1),  # raw logits (use BCEWithLogits)
        )

    def forward(self, sketch, image):
        return self.net(torch.cat([sketch, image], dim=1))


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


if __name__ == "__main__":
    g, d = UNetGenerator(), PatchDiscriminator()
    x = torch.randn(2, 3, 256, 256)
    y = g(x)
    print("G out:", y.shape, "| D out:", d(x, y).shape)
