import torch
import torch.nn as nn

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        super(DecoderBlock, self).__init__()

        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            nn.ReLU(),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            nn.ReLU(),
            nn.ReLU(),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            nn.BatchNorm2d(out_channels),
        )

        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding = 0)

    def forward(self, x):
        identity = self.skip(x)
        return identity + self.block(x)


class RUNet(nn.Module):

    def __init__(self, kernel_size=3, n_channels=64, padding = 1):
        super(RUNet, self).__init__()

        self.n = n_channels

        self.encoder = {
            1: nn.Sequential(
                nn.Conv2d(kernel_size, self.n, 7, padding=3),
                nn.BatchNorm2d(self.n),
                nn.ReLU(),
            ),
            2: nn.Sequential(
                ResidualBlock(self.n, self.n, kernel_size),
                ResidualBlock(self.n, self.n, kernel_size),
                ResidualBlock(self.n, self.n, kernel_size),
                ResidualBlock(self.n, self.n * 2, kernel_size),
            ),
            3: nn.Sequential(
                ResidualBlock(self.n * 2, self.n * 2, kernel_size),
                ResidualBlock(self.n * 2, self.n * 2, kernel_size),
                ResidualBlock(self.n * 2, self.n * 2, kernel_size),
                ResidualBlock(self.n * 2, self.n * 4, kernel_size),
            ),
            4: nn.Sequential(
                ResidualBlock(self.n * 4, self.n * 4, kernel_size),
                ResidualBlock(self.n * 4, self.n * 4, kernel_size),
                ResidualBlock(self.n * 4, self.n * 4, kernel_size),
                ResidualBlock(self.n * 4, self.n * 4, kernel_size),
                ResidualBlock(self.n * 4, self.n * 4, kernel_size),
                ResidualBlock(self.n * 4, self.n * 8, kernel_size),
            ),
            5: nn.Sequential(
                ResidualBlock(self.n * 8, self.n * 8, kernel_size),
                ResidualBlock(self.n * 8, self.n * 8, kernel_size),
                nn.BatchNorm2d(self.n * 8),
                nn.ReLU(),
            ),
            6: nn.Sequential(
                nn.Conv2d(self.n * 8, self.n * 16, kernel_size, padding = padding),
                nn.ReLU(),
            ),
        }

        self.decoder = {
            6: nn.Sequential(
                nn.Conv2d(self.n * 16, self.n * 8, kernel_size, padding = padding),
                nn.ReLU(),
                nn.PixelShuffle(1),
            ),
            5: DecoderBlock(self.n * 16, self.n * 8, kernel_size, padding),
            4: DecoderBlock(640, 384, kernel_size, padding),
            3: DecoderBlock(352, 256, kernel_size, padding),
            2: DecoderBlock(192, 96, kernel_size, padding),
            1: nn.Sequential(
                nn.Conv2d(88, 99, kernel_size, padding= padding),
                nn.ReLU(),
                nn.Conv2d(99, 99, kernel_size, padding= padding),
                nn.ReLU(),
                nn.Conv2d(99, kernel_size, 1, padding= padding),
            ),
        }

        self.pooling = nn.MaxPool2d((2, 2))

    def forward(self, x):
        stack = {}
        for name, layer in self.encoder.items():
            x = layer(x)

            if name <= 5:
                x = self.pooling(x)

            stack[name] = x

        for name, layer in self.decoder.items():
            if name == 6:
                x = layer(x)
                continue
            x = torch.cat((stack[name], x), 1)
            x = layer(x)
        return x
