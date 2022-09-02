from collections import OrderedDict

import torch
import torch.nn as nn


class conv3D(nn.Module):
    '3D convolutions. U-Net like model.'

    def __init__(self,
        num_classes: int = 2,
        in_channels=1, features=32):
        super(conv3D, self).__init__()

        # Leg 1
        self.encoder1 = conv3D._block(in_channels, features, name="enc1")
        self.encoder2 = conv3D._block(features, features * 2, name="enc2")
        self.encoder3 = conv3D._block(features * 2, features * 4, name="enc3")
        self.encoder4 = conv3D._block(features * 4, features * 8, name="enc4")

        self.bottleneck = conv3D._block(features * 8, features * 16, name="bottleneck")

        self.conv = nn.Conv3d(
            in_channels=features * 16, out_channels=1, kernel_size=1)

        # Pooling operation
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * 16 * 3, num_classes)

    def forward(self, x):
        # Leg 1
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        bottleneck = self.conv(self.bottleneck(self.pool(enc4)))
        bottleneck = self.relu(bottleneck)

        res = self.fc(torch.flatten(bottleneck, start_dim=1))

        return res

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm3d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
