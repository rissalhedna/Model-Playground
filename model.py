import torch
import torch.nn as nn
import torch.nn.functional as F
from parts import *


class LightweightClassifier(nn.Module):
    """
    Lightweight MobileNet-style classifier for image classification
    Optimized for 224x224 input and efficient inference
    """

    def __init__(self, n_channels=3, n_classes=4):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Initial conv layer
        self.stem = DSConv(n_channels, 32, 3, 2, 1)  # 224x224 -> 112x112

        # Feature extraction layers (MobileNet-style)
        self.features = nn.Sequential(
            # Stage 1: 112x112 -> 56x56
            DSBottleneck(32, 64, shortcut=False),
            DSConv(64, 64, 3, 2, 1),
            # Stage 2: 56x56 -> 28x28
            DSBottleneck(64, 128, shortcut=False),
            DSBottleneck(128, 128, shortcut=True),
            DSConv(128, 128, 3, 2, 1),
            # Stage 3: 28x28 -> 14x14
            DSBottleneck(128, 256, shortcut=False),
            DSBottleneck(256, 256, shortcut=True),
            DSBottleneck(256, 256, shortcut=True),
            DSConv(256, 256, 3, 2, 1),
            # Stage 4: 14x14 -> 7x7
            DSBottleneck(256, 512, shortcut=False),
            DSBottleneck(512, 512, shortcut=True),
            DSConv(512, 512, 3, 2, 1),
            # Final feature layer
            DSBottleneck(512, 1024, shortcut=False),
        )

        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(1024, n_classes)

        self._initialize_weights()

    def forward(self, x):
        # Stem
        x = self.stem(x)

        # Feature extraction
        x = self.features(x)

        # Global pooling
        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        # Classification
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
    """
    Ultra-lightweight classifier for edge deployment
    Even smaller and faster than LightweightClassifier
    """

    def __init__(self, n_channels=3, n_classes=4):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.features = nn.Sequential(
            # 224x224 -> 112x112
            DSConv(n_channels, 32, 3, 2, 1),
            # 112x112 -> 56x56
            DSBottleneck(32, 64, shortcut=False),
            DSConv(64, 64, 3, 2, 1),
            # 56x56 -> 28x28
            DSBottleneck(64, 128, shortcut=False),
            DSConv(128, 128, 3, 2, 1),
            # 28x28 -> 14x14
            DSBottleneck(128, 256, shortcut=False),
            DSConv(256, 256, 3, 2, 1),
            # 14x14 -> 7x7
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
    """
    Create a lightweight image classifier

    Args:
        input_channels: Number of input channels (3 for RGB)
        num_classes: Number of classification classes

    Returns:
        LightweightClassifier model ready for training
    """
    return LightweightClassifier(input_channels, num_classes)


def create_tiny_classifier(input_channels=3, num_classes=4):
    """
    Create an ultra-lightweight classifier for edge deployment

    Args:
        input_channels: Number of input channels (3 for RGB)
        num_classes: Number of classification classes

    Returns:
        TinyClassifier model ready for training
    """
    return TinyClassifier(input_channels, num_classes)


# Legacy functions for backward compatibility
def create_lightweight_detector(input_channels=3, num_classes=4):
    return create_lightweight_classifier(input_channels, num_classes)


def create_nano_detector(input_channels=3, num_classes=4):
    return create_tiny_classifier(input_channels, num_classes)
