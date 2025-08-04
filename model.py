import torch
import torch.nn as nn
import torch.nn.functional as F
from parts import *


class LightweightClassifier(nn.Module):

    def __init__(self, n_channels=3, n_classes=4):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.stem = DSConv(n_channels, 32, 3, 2, 1)

        self.features = nn.Sequential(
            DSBottleneck(32, 64, shortcut=False),
            DSConv(64, 64, 3, 2, 1),
            DSBottleneck(64, 128, shortcut=False),
            DSBottleneck(128, 128, shortcut=True),
            DSConv(128, 128, 3, 2, 1),
            DSBottleneck(128, 256, shortcut=False),
            DSBottleneck(256, 256, shortcut=True),
            DSBottleneck(256, 256, shortcut=True),
            DSConv(256, 256, 3, 2, 1),
            DSBottleneck(256, 512, shortcut=False),
            DSBottleneck(512, 512, shortcut=True),
            DSConv(512, 512, 3, 2, 1),
            DSBottleneck(512, 1024, shortcut=False),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(1024, n_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)

        x = self.features(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        x = self.dropout(x)
        x = self.classifier(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class TinyClassifier(nn.Module):

    def __init__(self, n_channels=3, n_classes=4):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.features = nn.Sequential(
            DSConv(n_channels, 32, 3, 2, 1),
            DSBottleneck(32, 64, shortcut=False),
            DSConv(64, 64, 3, 2, 1),
            DSBottleneck(64, 128, shortcut=False),
            DSConv(128, 128, 3, 2, 1),
            DSBottleneck(128, 256, shortcut=False),
            DSConv(256, 256, 3, 2, 1),
            DSBottleneck(256, 512, shortcut=False),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(512, n_classes))

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.down1 = down(n_channels, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)

        self.up1 = up(256, 128)
        self.up2 = up(128, 64)
        self.up3 = up(64, 32)
        self.fc = nn.Sequential(
            nn.Linear(18432, 24),
            nn.Linear(24, n_classes),
        )

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def create_lightweight_classifier(input_channels=3, num_classes=4):
    return LightweightClassifier(input_channels, num_classes)


def create_tiny_classifier(input_channels=3, num_classes=4):
    return TinyClassifier(input_channels, num_classes)


def create_lightweight_detector(input_channels=3, num_classes=4):
    return create_lightweight_classifier(input_channels, num_classes)


def create_nano_detector(input_channels=3, num_classes=4):
    return create_tiny_classifier(input_channels, num_classes)
