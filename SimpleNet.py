import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm3d(20),
            nn.MaxPool3d(kernel_size=5, stride=2, padding=2)
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
        x = self.conv1(x)
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
            # nn.Dropout3d(0.1)
        )

    def forward(self, x):
        x1 = self.net(x)
        x = torch.cat((x, x1), dim=1)
        return x


class DenseSharp(nn.Module):
    def __init__(self):
        super(DenseSharp, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            # nn.Dropout3d(0.1)
        )
        self.denseblock1 = nn.ModuleList([DenseBlockLayer(32 + i * 16) for i in range(4)])
        self.trans1 = nn.Sequential(
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.Conv3d(96, 48, kernel_size=1),
            nn.AvgPool3d(kernel_size=2),
            # nn.Dropout3d(0.2)
        )
        self.denseblock2 = nn.ModuleList([DenseBlockLayer(48 + i * 16) for i in range(4)])
        self.trans2 = nn.Sequential(
            nn.BatchNorm3d(112),
            nn.ReLU(inplace=True),
            nn.Conv3d(112, 56, kernel_size=1),
            nn.AvgPool3d(kernel_size=2),
            # nn.Dropout3d(0.2)
        )
        self.denseblock3 = nn.ModuleList([DenseBlockLayer(56 + i * 16) for i in range(4)])
        self.trans3 = nn.Sequential(
            nn.BatchNorm3d(120),
            nn.ReLU(inplace=True),
            nn.Conv3d(120, 60, kernel_size=1),
            nn.AvgPool3d(kernel_size=2),
            # nn.Dropout3d(0.2)
        )
        self.denseblock4 = nn.ModuleList([DenseBlockLayer(60 + i * 16) for i in range(4)])
        self.trans4 = nn.Sequential(
            nn.BatchNorm3d(124),
            nn.ReLU(inplace=True),
            nn.AvgPool3d(kernel_size=4)
        )
        self.output = nn.Sequential(
            nn.Dropout(0.81),
            nn.Linear(124, 1),
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
        for i in range(4):
            x = self.denseblock4[i](x)
        x = self.trans4(x)
        x = x.view(x.shape[0], -1).squeeze()
        x = self.output(x)
        return x.squeeze()


class SasakiNet(nn.Module):
    def __init__(self):
        super(SasakiNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.Sequential(
            nn.AvgPool3d(kernel_size=8)
        )
        self.output = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = x.view(x.shape[0], -1).squeeze()
        x = self.output(x)
        return x.squeeze()


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, padding=1),
        )
        self.res1 = nn.Sequential(
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, kernel_size=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, padding=1)
        )
        self.res2 = nn.Sequential(
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, kernel_size=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, padding=1)
        )
        self.res3 = nn.Sequential(
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, kernel_size=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, padding=1)
        )
        self.res4 = nn.Sequential(
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, kernel_size=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, padding=1)
        )
        self.reduce1 = nn.Sequential(
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1)
        )
        self.res5 = nn.Sequential(
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 64, kernel_size=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
        )
        self.res6 = nn.Sequential(
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 64, kernel_size=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
        )
        self.res7 = nn.Sequential(
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 64, kernel_size=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
        )
        self.res8 = nn.Sequential(
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 64, kernel_size=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
        )
        self.reduce2 = nn.Sequential(
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(128, 256, kernel_size=3, padding=1)
        )
        self.res9 = nn.Sequential(
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 128, kernel_size=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, kernel_size=3, padding=1)
        )
        self.res10 = nn.Sequential(
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 128, kernel_size=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, kernel_size=3, padding=1)
        )
        self.res11 = nn.Sequential(
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 128, kernel_size=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, kernel_size=3, padding=1)
        )
        self.res12 = nn.Sequential(
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 128, kernel_size=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, kernel_size=3, padding=1)
        )
        self.reduce3 = nn.Sequential(
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.AvgPool3d(kernel_size=8)
        )
        self.output = nn.Sequential(
            nn.Dropout(0.75),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.res1(x)
        x = x + x1
        x1 = self.res2(x)
        x = x + x1
        x1 = self.res3(x)
        x = x + x1
        x1 = self.res4(x)
        x = x + x1
        x = self.reduce1(x)
        x1 = self.res5(x)
        x = x + x1
        x1 = self.res6(x)
        x = x + x1
        x1 = self.res7(x)
        x = x + x1
        x1 = self.res8(x)
        x = x + x1
        x = self.reduce2(x)
        x1 = self.res9(x)
        x = x + x1
        x1 = self.res10(x)
        x = x + x1
        x1 = self.res11(x)
        x = x + x1
        x1 = self.res12(x)
        x = x + x1
        x = self.reduce3(x)
        x = x.view(x.shape[0], -1)
        x = self.output(x)
        return x.squeeze()


#  net = ResNet().cuda()
# print(net)
# x = torch.randn(10, 1, 32, 32, 32).cuda()
# net(x)
