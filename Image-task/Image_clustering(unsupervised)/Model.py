# autoencoder模型
import torch
import torch.nn as nn
import numpy as np


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        # encoder
        self.cov1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.cov2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.cov3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)
        self.relu = nn.ReLU()

        # decoder
        self.unconv1 = nn.ConvTranspose2d(256, 128, 3, 1, 1)
        self.unconv2 = nn.ConvTranspose2d(128, 64, 3, 1, 1)
        self.unconv3 = nn.ConvTranspose2d(64, 3, 3, 1, 1)
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # encoder
        x = self.cov1(x)  # [64,32,32]
        x = self.relu(x)
        x, indicate_1 = self.pool(x)  # [64,16,16]

        x = self.cov2(x)  # [128,16,16]
        x = self.relu(x)
        x, indicate_2 = self.pool(x)  # [128,8,8]

        x = self.cov3(x)  # [256,8,8]
        x = self.relu(x)
        x, indicate_3 = self.pool(x)  # [256,4,4]

        # decoder
        x1 = self.unpool(x, indicate_3)  # [256,8,8]
        x1 = self.unconv1(x1)  # [128,8,8]
        x1 = self.relu(x1)

        x1 = self.unpool(x1, indicate_2)  # [128,16,16]
        x1 = self.unconv2(x1)  # [64,16,16]
        x1 = self.relu(x1)

        x1 = self.unpool(x1, indicate_1)  # [64,32,32]
        x1 = self.unconv3(x1)  # [3,32,32]
        x1 = self.tanh(x1)

        return x, x1


