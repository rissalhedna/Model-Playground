import torch
import torch.nn as nn
import torch.nn.functional as F


def down(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(kernel_size=2),
        nn.ReLU(),
    )


def up(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


class DSConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class DSBottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, shortcut=True):
        super().__init__()
        self.conv1 = DSConv(in_channels, out_channels, 3, 1, 1)
        self.conv2 = DSConv(out_channels, out_channels, 3, 1, 1)
        self.shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        if self.shortcut:
            return x + self.conv2(self.conv1(x))
        return self.conv2(self.conv1(x))


class C3DSBlock(nn.Module):

    def __init__(self, in_channels, out_channels, n=1, shortcut=True):
        super().__init__()
        self.cv1 = DSConv(in_channels, out_channels // 2, 1, 1, 0)
        self.cv2 = DSConv(in_channels, out_channels // 2, 1, 1, 0)
        self.cv3 = DSConv(out_channels, out_channels, 1, 1, 0)
        self.m = nn.Sequential(
            *(
                DSBottleneck(out_channels // 2, out_channels // 2, shortcut)
                for _ in range(n)
            )
        )

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPF(nn.Module):

    def __init__(self, in_channels, out_channels, k=5):
        super().__init__()
        c_ = in_channels // 2
        self.cv1 = DSConv(in_channels, c_, 1, 1, 0)
        self.cv2 = DSConv(c_ * 4, out_channels, 1, 1, 0)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))
