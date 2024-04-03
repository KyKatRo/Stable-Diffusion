import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue = x

        n, c, h, w = x.shape

        x = x.view(n, c, h * w)

        x = x.transpose(-1, -2)

        x = self.attention(x)

        x = x.transpose(-1, -2)

        x = x.view(n, c, h, w)

        return residue + x


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # If the number of input channels is the same as the number of output channels, we don't need to do anything
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue = x

        x = self.groupnorm1(x)
        x = F.silu(x)

        x = self.conv1(x)
        x = self.groupnorm2(x)

        x = F.silu(x)

        x = self.conv2(x)

        return self.residual(residue) + x


class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2D(4, 4, kernel_size=1, padding=0),
            nn.Conv2D(4, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),

            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            nn.Upsample(scale_factor=2),

            nn.Con
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self:
            if isinstance(module, nn.ConvTranspose2d):
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        return x