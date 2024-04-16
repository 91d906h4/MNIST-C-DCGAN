import torch

from torch import nn


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.block1 = self.block(in_channels=1, out_channels=64, kernel_size=3)
        self.block2 = self.block(in_channels=64, out_channels=128, kernel_size=3)
        self.block3 = self.block(in_channels=128, out_channels=256, kernel_size=3)

        self.linear1 = nn.Linear(2304, 128)
        self.linear2 = nn.Linear(128, 1)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()

    def block(self, in_channels: int, out_channels: int, kernel_size: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.flatten(x)

        x = self.linear1(x)
        x = self.relu(x)

        x = self.linear2(x)
        x = self.sigmoid(x)

        return x


class Generator(nn.Module):
    def __init__(self) -> None:
        super(Generator, self).__init__()
        self.block1 = self.block(in_channels=1, out_channels=64, kernel_size=3)
        self.conv1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2)
        self.block2 = self.block(in_channels=64, out_channels=128, kernel_size=3)
        self.conv2 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2)
        self.block3 = self.block(in_channels=128, out_channels=256, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=2, padding="same")

        self.tanh = nn.Tanh()


    def block(self, in_channels: int, out_channels: int, kernel_size: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.conv1(x)
        x = self.block2(x)
        x = self.conv2(x)
        x = self.block3(x)
        x = self.conv3(x)

        x = self.tanh(x)

        return x