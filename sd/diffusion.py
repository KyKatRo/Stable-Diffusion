import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):
    def __init__(self, n_embed: int):
        super().__init__()

        self.linear1 = nn.Linear(n_embed, n_embed * 4)
        self.linear2 = nn.Linear(n_embed * 4, n_embed * 4)

    def forward(self, time: torch.LongTensor) -> torch.Tensor:
        x = self.linear1(time)

        x = F.silu(x)

        x = self.linear2(x)

        return x


class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_time=1280):
        super().__init__()

        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv1_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv2_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, time: torch.LongTensor) -> torch.Tensor:
        residue = x

        x = self.groupnorm_feature(x)
        x = F.silu(x)

        x = self.conv1_feature(x)

        time = self.linear_time(time)

        merged = x + time.unsqueeze(-1).unsqueeze(-1)

        merged = self.groupnorm_merged(merged)

        merged = F.silu(merged)

        merged = self.conv2_merged(merged)

        return self.residual(residue) + merged


class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_heads: int, n_embed: int, d_context=768):
        super().__init__()

        channels = n_embed * n_heads

        self.groupnorm = nn.GroupNorm(32, channels)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layer_norm1 = nn.LayerNorm(channels)
        self.attention1 = SelfAttention(n_heads, channels, in_proj_bias=False)
        self.layer_norm2 = nn.LayerNorm(channels)
        self.attention2 = CrossAttention(n_heads, channels, d_context, in_proj_bias=False)
        self.layer_norm3 = nn.LayerNorm(channels)

        self.linear_geglu_1 = nn.Linear(channels, channels * 4)
        self.linear_geglu_2 = nn.Linear(channels * 4, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        residue_long = x

        x = self.groupnorm(x)
        x = self.conv_input(x)

        n, c, h, w = x.shape
        x = x.view(n, c, h * w)
        x = x.transpose(-1, -2)

        residue_short = x

        x = self.layer_norm1(x)
        self.attention1(x, context)
        x += residue_short

        residue_short = x

        x = self.layer_norm2(x)
        self.attention2(x, context)

        x += residue_short

        residue_short = x

        x = self.layer_norm3(x)

        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x *= F.gelu(gate)

        x = self.linear_geglu_2(x)

        x += residue_short

        x = x.transpose(-1, -2)
        x = x.view(n, c, h, w)

        return self.conv_output(x) + residue_long


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")

        return self.conv(x)


class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.groupnorm(x)
        x = F.silu(x)

        x = self.conv1(x)

        return x


class SwitchSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.LongTensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)

        return x


class UNET(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoders = nn.ModuleList([
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 320)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 320)),

        ])

        self.bottle_neck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),
            UNET_AttentionBlock(8, 160),
            UNET_ResidualBlock(1280, 1280)
        )

        self.decoders = nn.ModuleList([
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)),

            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(1280, 320), UNET_AttentionBlock(8, 40), Upsample(640)),

            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
        ])


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.LongTensor) -> torch.Tensor:
        time = self.time_embedding(time)

        output = self.unet(latent, context, time)

        return self.final(output)
