import torch.nn as nn


class Reshape(nn.Module):
    """Custom reshape class for use in Sequential."""

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class Generator(nn.Module):
    def __init__(self, noise_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 512 * 4 * 4),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.ReLU(inplace=True),
            Reshape((-1, 512, 4, 4)),
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),  # double size of image to 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),  # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),  # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=3,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),  # 64x64
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)
