import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channel, f, filters,groups,s):
        super(ConvBlock, self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.Conv2d(in_channel, F1, 1, stride=s, padding=0, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(True),
            nn.Conv2d(F1, F2, f, stride=1, padding=True, bias=False, groups=groups),
            nn.BatchNorm2d(F2),
            nn.ReLU(True),
            nn.Conv2d(F2, F3, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F3),
        )
        self.shortcut_1 = nn.Conv2d(in_channel, F3, 1, stride=s, padding=0, bias=False)
        self.batch_1 = nn.BatchNorm2d(F3)
        self.relu_1 = nn.ReLU(True)

    def forward(self, X):
        X_shortcut = self.shortcut_1(X)
        X_shortcut = self.batch_1(X_shortcut)
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X


class IndentityBlock(nn.Module):
    def __init__(self, in_channel, f, filters, groups):
        super(IndentityBlock, self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.Conv2d(in_channel, F1, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(True),
            nn.Conv2d(F1, F2, f, stride=1, padding=True, bias=False, groups=groups),
            nn.BatchNorm2d(F2),
            nn.ReLU(True),
            nn.Conv2d(F2, F3, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F3),
        )
        self.relu_1 = nn.ReLU(True)

    def forward(self, X):
        X_shortcut = X
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X


class ResNet50(nn.Module):
    def __init__(self, in_channels, n_class):
        super(ResNet50, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, padding=1),
        )
        self.stage2 = nn.Sequential(
            ConvBlock(64, f=3, filters=[128, 128, 256], groups=32, s=1),
            IndentityBlock(256, 3, [128, 128, 256], groups=32),
            IndentityBlock(256, 3, [128, 128, 256], groups=32),
        )
        self.stage3 = nn.Sequential(
            ConvBlock(256, f=3, filters=[256, 256, 512], groups=32, s=2),
            IndentityBlock(512, 3, [256, 256, 512], groups=32),
            IndentityBlock(512, 3, [256, 256, 512], groups=32),
            IndentityBlock(512, 3, [256, 256, 512], groups=32),
        )
        self.stage4 = nn.Sequential(
            ConvBlock(512, f=3, filters=[512, 512, 1024], groups=32,s=2),
            IndentityBlock(1024, 3, [512, 512, 1024], groups=32),
            IndentityBlock(1024, 3, [512, 512, 1024], groups=32),
            IndentityBlock(1024, 3, [512, 512, 1024], groups=32),
            IndentityBlock(1024, 3, [512, 512, 1024], groups=32),
            IndentityBlock(1024, 3, [512, 512, 1024], groups=32),
        )
        self.stage5 = nn.Sequential(
            ConvBlock(1024, f=3, filters=[1024, 1024, 2048], groups=32, s=2),
            IndentityBlock(2048, 3, [1024, 1024, 2048], groups=32),
            IndentityBlock(2048, 3, [1024, 1024, 2048], groups=32),
        )
        self.pool = nn.AvgPool2d(2, 2, padding=1)
        self.fc = nn.Sequential(
            nn.Linear(2048, n_class)
        )

    def forward(self, X):
        out = self.stage1(X)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.stage5(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out