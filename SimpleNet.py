import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 10, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm3d(10),
            nn.MaxPool3d(kernel_size=5, stride=2, padding=2)
        )
        self.conv11 = nn.Sequential(
            nn.Conv3d(1, 10, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm3d(10),
            nn.MaxPool3d(kernel_size=3, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(20, 40, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm3d(40),
            nn.MaxPool3d(kernel_size=3)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(40, 60, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm3d(60),
            nn.MaxPool3d(kernel_size=3)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(60, 30, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(30),
            nn.AvgPool3d(kernel_size=3, padding=1)
        )
        self.classify = nn.Sequential(
            nn.Linear(3840, 1920),
            nn.ReLU(inplace=True),
            nn.Linear(1920, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.conv11(x)
        x = self.conv1(x)
        x = torch.cat((x1, x), dim=1)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        x = self.classify(x)
        return x.squeeze()


class CropNet(nn.Module):
    def __init__(self):
        super(CropNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(32),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(128),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.output = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(128),
            nn.AvgPool3d(kernel_size=3, stride=1),
            nn.Conv3d(128, 1, kernel_size=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.output(x)
        return x.squeeze()


class DenseBlockLayer(nn.Module):
    def __init__(self, input_channel, filters=16, bottleneck=4):
        super(DenseBlockLayer, self).__init__()
        self.net = nn.Sequential(
            nn.BatchNorm3d(input_channel),
            nn.ReLU(inplace=True),
            nn.Conv3d(input_channel, filters * bottleneck, kernel_size=1),
            nn.BatchNorm3d(filters * bottleneck),
            nn.ReLU(inplace=True),
            nn.Conv3d(filters * bottleneck, filters, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x1 = self.net(x)
        x = torch.cat((x, x1), dim=1)
        return x


class DenseSharp(nn.Module):
    def __init__(self):
        super(DenseSharp, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.denseblock1 = nn.ModuleList([DenseBlockLayer(32 + i * 16) for i in range(4)])
        self.trans1 = nn.Sequential(
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.Conv3d(96, 48, kernel_size=1, padding=1),
            nn.AvgPool3d(kernel_size=2)
        )
        self.denseblock2 = nn.ModuleList([DenseBlockLayer(48 + i * 16) for i in range(4)])
        self.trans2 = nn.Sequential(
            nn.BatchNorm3d(112),
            nn.ReLU(inplace=True),
            nn.Conv3d(112, 56, kernel_size=1, padding=1),
            nn.AvgPool3d(kernel_size=2)
        )
        self.denseblock3 = nn.ModuleList([DenseBlockLayer(56 + i * 16) for i in range(4)])
        self.trans3 = nn.Sequential(
            nn.BatchNorm3d(120),
            nn.ReLU(inplace=True),
            nn.AvgPool3d(kernel_size=8)
        )
        self.output = nn.Sequential(
            nn.Linear(120, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        for i in range(4):
            x = self.denseblock1[i](x)
        x = self.trans1(x)
        for i in range(4):
            x = self.denseblock2[i](x)
        x = self.trans2(x)
        for i in range(4):
            x = self.denseblock3[i](x)
        x = self.trans3(x)
        x = x.view(x.shape[0], -1).squeeze()
        x = self.output(x)
        return x.squeeze()



