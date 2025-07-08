import torch
import torch.nn as nn
import torch.nn.functional as F

def down(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(kernel_size=2),
        # nn.ReLU(),
    )

def up(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.BatchNorm2d(out_channels),
        # nn.ReLU(),
    )
