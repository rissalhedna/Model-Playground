import torch
import torch.nn as nn
import torch.nn.functional as F
from parts import *

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
            nn.Linear(24, 10),
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