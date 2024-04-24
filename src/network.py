import torch

from torch import nn


class Discriminator(nn.Module):
    def __init__(self, image_size: int) -> None:
        super(Discriminator, self).__init__()
        self.block1 = self.block(in_channels=2, out_channels=64, kernel_size=4)
        self.block2 = self.block(in_channels=64, out_channels=128, kernel_size=4)
        self.block3 = self.block(in_channels=128, out_channels=256, kernel_size=3)
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, stride=1, padding=0)

        # We have 10 labels (0 ~ 9), so the number of input embedding is 10.
        self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=image_size**2)

        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def block(in_channels: int, out_channels: int, kernel_size: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding="same", bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y = self.embedding(y) # 1 -> 784
        y = y.view(-1, 1, 28, 28) # 784 -> 1 x 28 x 28

        # Concatenate image x and label y, make x a
        # 2 channels input.
        x = torch.cat([x, y], dim=1)

        x = self.block1(x) # 2 x 28 x 28 -> 64 x 14 x 14
        x = self.block2(x) # 64 x 14 x 14 -> 128 x 7 x 7
        x = self.block3(x) # 128 x 7 x 7 -> 256 x 3 x 3

        x = self.conv1(x) # 256 x 3 x 3 -> 1 x 1 x 1
        x = self.flatten(x) # 1 x 1 x 1 -> 1

        x = self.sigmoid(x)

        return x


class Generator(nn.Module):
    def __init__(self, z_dim: int) -> None:
        super(Generator, self).__init__()
        self.block1 = self.block(in_channels=z_dim, out_channels=256, kernel_size=4, stride=1, padding=0)
        self.block2 = self.block(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.block3 = self.block(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=3, bias=False)

        # We have 10 labels (0 ~ 9), so the number of input embedding is 10.
        self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=z_dim)

        self.tanh = nn.Tanh()

    @staticmethod
    def block(in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y = self.embedding(y) # 1 -> z_dim
        y = y.view(-1, 100, 1, 1) # z_dim -> z_dim x 1 x 1

        # Multiply noise input x and label y.
        x = x * y

        x = self.block1(x) # z_dim x 1 x 1 -> 256 x 4 x 4
        x = self.block2(x) # 256 x 4 x 4 -> 128 x 8 x 8
        x = self.block3(x) # 128 x 8 x 8 -> 64 x 16 x 16

        x = self.conv1(x) # 64 x 16 x 16 -> 1 x 28 x 28

        x = self.tanh(x)

        return x