import torch
import torch.nn as nn
import torch.nn.functional as F

class DSConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
    ):
        super(DSConv2d, self).__init__()
        self.blocks = nn.Sequential(
            DSConv2d(
                in_channels,
                in_channels,
                stride=stride,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            ),
            DSConv2d(
                in_channels, out_channels, stride=stride, kernel_size=1, bias=bias
            ),
        )

    def forward(self, x):
        return self.blocks(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        super(DecoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            DSConv2d(
                in_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            nn.LeakyReLU(),
            DSConv2d(
                out_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.block(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            DSConv2d(
                in_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            DSConv2d(
                out_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            nn.BatchNorm2d(out_channels),
        )

        self.skip = DSConv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        identity = self.skip(x)
        return identity + self.block(x)

def init_weights(m):
    if isinstance(m, DSConv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.Sequential):
        for layer in m:
            init_weights(layer)

    elif isinstance(m, ResidualBlock) or isinstance(m, DecoderBlock):
        for layer in m.block:
            init_weights(layer)

    elif isinstance(m, AttentionGate):
        for layer in m.W_g:
            init_weights(layer)
        for layer in m.W_x:
            init_weights(layer)
        for layer in m.psi:
            init_weights(layer)


class RUNet(nn.Module):
    def __init__(self, kernel_size=3, n_channels=64, padding=1):
        super(RUNet, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"

        self.n = n_channels

        self.encoder = nn.ModuleDict(
            {
                "1": nn.Sequential(
                    DSConv2d(kernel_size, self.n, 7, padding=3),
                    nn.BatchNorm2d(self.n),
                    nn.LeakyReLU(),
                ),
                "2": nn.Sequential(
                    ResidualBlock(self.n, self.n, kernel_size),
                    ResidualBlock(self.n, self.n, kernel_size),
                    ResidualBlock(self.n, self.n, kernel_size),
                    ResidualBlock(self.n, self.n * 2, kernel_size),
                ),
                "3": nn.Sequential(
                    ResidualBlock(self.n * 2, self.n * 2, kernel_size),
                    ResidualBlock(self.n * 2, self.n * 2, kernel_size),
                    ResidualBlock(self.n * 2, self.n * 2, kernel_size),
                    ResidualBlock(self.n * 2, self.n * 4, kernel_size),
                ),
                "4": nn.Sequential(
                    ResidualBlock(self.n * 4, self.n * 4, kernel_size),
                    ResidualBlock(self.n * 4, self.n * 4, kernel_size),
                    ResidualBlock(self.n * 4, self.n * 4, kernel_size),
                    ResidualBlock(self.n * 4, self.n * 4, kernel_size),
                    ResidualBlock(self.n * 4, self.n * 4, kernel_size),
                    ResidualBlock(self.n * 4, self.n * 8, kernel_size),
                ),
                "5": nn.Sequential(
                    ResidualBlock(self.n * 8, self.n * 8, kernel_size),
                    ResidualBlock(self.n * 8, self.n * 8, kernel_size),
                    nn.BatchNorm2d(self.n * 8),
                    nn.LeakyReLU(),
                ),
                "6": nn.Sequential(
                    DSConv2d(self.n * 8, self.n * 16, kernel_size, padding=padding),
                    nn.LeakyReLU(),
                ),
            }
        )

        self.decoder = nn.ModuleDict(
            {
                "6": nn.Sequential(
                    DSConv2d(self.n * 16, self.n * 8, kernel_size, padding=padding),
                    nn.LeakyReLU(),
                ),
                "5": DecoderBlock(self.n * 16, self.n * 8, kernel_size, padding),
                "4": DecoderBlock(640, 384, kernel_size, padding),
                "3": DecoderBlock(352, 256, kernel_size, padding),
                "2": DecoderBlock(192, 96, kernel_size, padding),
                "1": nn.Sequential(
                    #nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True),
                    DSConv2d(88, 99, kernel_size, padding=padding),
                    nn.LeakyReLU(),
                    DSConv2d(99, 99, kernel_size, padding=padding),
                    nn.LeakyReLU(),
                    DSConv2d(99, kernel_size, 1, padding=0),
                )
            }
        )

        self.pooling = nn.MaxPool2d((2, 2))

        self.pixelshuffle = nn.PixelShuffle(2)

    def forward(self, x):
        stack = {}

        for name, layer in self.encoder.items():
            x = layer(x)
            if name <= "5":
                x = self.pooling(x)
            stack[name] = x

        for name, layer in self.decoder.items():
            if name == "6":
                x = layer(x)
                
                continue
            
            x = torch.cat([stack[name], x], dim=1)
            
            x = layer(x)
            
            if name == "1":
                break

            x = self.pixelshuffle(x)
        return x
