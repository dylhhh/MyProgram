import torch.nn as nn
import torch.nn.functional as F


# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, strides=1):
        super(ResidualBlock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=strides, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        if in_channels != out_channels or strides != 1:
            self.layer2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides),
                nn.BatchNorm2d(out_channels))
        else:
            self.layer = nn.Sequential()

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(x)
        out = F.relu(out1 + out2)
        return out


class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        # [3, 128, 128]
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 1, 1)  # [64, 128, 128]
        )
        self.layer1 = ResidualBlock(64, 128, 2)  # [128, 64, 64]
        self.layer2 = ResidualBlock(128, 256, 2)  # [256, 32, 32]
        self.layer3 = ResidualBlock(256, 512, 2)  # [512, 16, 16]
        self.layer4 = ResidualBlock(512, 1024, 2)  # [1024, 8, 8]
        self.pool = nn.AvgPool2d(2)  # [1024, 4, 4]
        self.fc = nn.Sequential(
            nn.Linear(1024 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out).view(out.size()[0], -1)
        out = self.fc(out)
        return out
