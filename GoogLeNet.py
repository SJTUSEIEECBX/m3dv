import torch
import torch.nn as nn


class Stem(nn.Module):
    def __init__(self):
        super(Stem, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3)
        self.conv4 = nn.Conv3d(64, 96, kernel_size=3, stride=2)
        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=2)
        self.conv5 = nn.Conv3d(160, 64, kernel_size=1)
        self.conv6 = nn.Conv3d(64, 96, kernel_size=3, padding=1)
        self.conv7 = nn.Conv3d(160, 64, kernel_size=1)
        self.conv8 = nn.Conv3d(64, 64, kernel_size=[5, 1, 1], padding=[2, 0, 0])
        self.conv9 = nn.Conv3d(64, 64, kernel_size=[1, 5, 1], padding=[0, 2, 0])
        self.conv10 = nn.Conv3d(64, 64, kernel_size=[1, 1, 5], padding=[0, 0, 2])
        self.conv11 = nn.Conv3d(64, 96, kernel_size=3, padding=1)
        self.conv12 = nn.Conv3d(192, 192, kernel_size=3)
        self.pool2 = nn.MaxPool3d(kernel_size=3, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x1 = self.conv4(x)
        x2 = self.pool1(x)
        x = torch.cat((x1, x2), dim=1)
        x1 = self.conv5(x)
        x1 = self.conv6(x1)
        x2 = self.conv7(x)
        x2 = self.conv8(x2)
        x2 = self.conv9(x2)
        x2 = self.conv10(x2)
        x2 = self.conv11(x2)
        x = torch.cat((x1, x2), dim=1)
        x1 = self.conv12(x)
        x2 = self.pool2(x)
        x = torch.cat((x1, x2), dim=1)
        return x


class InceptionA(nn.Module):
    def __init__(self):
        super(InceptionA, self).__init__()
        self.conv1 = nn.Conv3d(384, 32, kernel_size=1)
        self.conv2 = nn.Conv3d(384, 32, kernel_size=1)
        self.conv3 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(384, 32, kernel_size=1)
        self.conv5 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.conv6 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.conv7 = nn.Conv3d(96, 384, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x2 = self.conv3(x2)
        x3 = self.conv4(x)
        x3 = self.conv5(x3)
        x3 = self.conv6(x3)
        x4 = torch.cat((x1, x2, x3), dim=1)
        x4 = self.conv7(x4)
        x = x + x4
        return x


class ReductionA(nn.Module):
    def __init__(self):
        super(ReductionA, self).__init__()
        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=2)
        self.conv1 = nn.Conv3d(384, 384, kernel_size=3, stride=2)
        self.conv2 = nn.Conv3d(384, 256, kernel_size=1)
        self.conv3 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(256, 384, kernel_size=3, stride=2)

    def forward(self, x):
        x1 = self.pool1(x)
        x2 = self.conv1(x)
        x3 = self.conv2(x)
        x3 = self.conv3(x3)
        x3 = self.conv4(x3)
        x = torch.cat((x1, x2, x3), dim=1)
        return x


class InceptionB(nn.Module):
    def __init__(self):
        super(InceptionB, self).__init__()
        self.conv1 = nn.Conv3d(1152, 192, kernel_size=1)
        self.conv2 = nn.Conv3d(1152, 128, kernel_size=1)
        self.conv3 = nn.Conv3d(128, 192, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(384, 1152, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x2 = self.conv3(x2)
        x3 = torch.cat((x1, x2), dim=1)
        x3 = self.conv4(x3)
        x = x + x3
        return x


class ReductionB(nn.Module):
    def __init__(self):
        super(ReductionB, self).__init__()
        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=2)
        self.conv1 = nn.Conv3d(1152, 256, kernel_size=1)
        self.conv2 = nn.Conv3d(256, 384, kernel_size=3, stride=2)
        self.conv3 = nn.Conv3d(1152, 256, kernel_size=1)
        self.conv4 = nn.Conv3d(256, 320, kernel_size=3, padding=1)
        self.conv5 = nn.Conv3d(320, 384, kernel_size=3, stride=2)

    def forward(self, x):
        x1 = self.pool1(x)
        x2 = self.conv1(x)
        x2 = self.conv2(x2)
        x3 = self.conv3(x)
        x3 = self.conv4(x3)
        x3 = self.conv5(x3)
        x = torch.cat((x1, x2, x3), dim=1)
        return x


class InceptionC(nn.Module):
    def __init__(self):
        super(InceptionC, self).__init__()
        self.conv1 = nn.Conv3d(1920, 192, kernel_size=1)
        self.conv2 = nn.Conv3d(1920, 128, kernel_size=1)
        self.conv3 = nn.Conv3d(128, 192, kernel_size=[3, 1, 1], padding=[1, 0, 0])
        self.conv4 = nn.Conv3d(192, 224, kernel_size=[1, 3, 1], padding=[0, 1, 0])
        self.conv5 = nn.Conv3d(224, 256, kernel_size=[1, 1, 3], padding=[0, 0, 1])
        self.conv6 = nn.Conv3d(448, 1920, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x2 = self.conv3(x2)
        x2 = self.conv4(x2)
        x2 = self.conv5(x2)
        x3 = torch.cat((x1, x2), dim=1)
        x3 = self.conv6(x3)
        x = x + x3
        return x


class GoogLeNet(nn.Module):
    def __init__(self, layerA=5, layerB=10, layerC=5):
        super(GoogLeNet, self).__init__()
        self.layerA = layerA
        self.layerB = layerB
        self.layerC = layerC
        self.stem = Stem()
        self.inceptionA = nn.ModuleList([InceptionA() for i in range(layerA)])
        self.reductionA = ReductionA()
        self.inceptionB = nn.ModuleList([InceptionB() for i in range(layerB)])
        self.reductionB = ReductionB()
        self.inceptionC = nn.ModuleList([InceptionC() for i in range(layerC)])
        self.avgpool = nn.AvgPool3d(kernel_size=4)
        self.dropout = nn.Dropout(0.8, inplace=True)
        self.output = nn.Sequential(
            nn.Conv3d(1920, 512, kernel_size=1),
            nn.Conv3d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.stem(x)
        for i in range(self.layerA):
            x = self.inceptionA[i](x)
        x = self.reductionA(x)
        # x1 = x.clone()
        for i in range(self.layerB):
            x = self.inceptionB[i](x)
        x = self.reductionB(x)
        # x2 = x.clone()
        for i in range(self.layerC):
            x = self.inceptionC[i](x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = self.output(x)
        x = x.squeeze()
        return x


if __name__ == 'main':
    net = GoogLeNet().cuda()
    print('hello')
