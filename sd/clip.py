import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class CLIPEmbedding(nn.Module):
    def __init__(self, vocab_size: int, n_embed: int, n_special_tokens: int):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.position_embedding = nn.Parameter(torch.zeros(n_special_tokens, n_embed))

    def forward(self, tokens: torch.LongTensor) -> torch.Tensor:
        x = self.token_embedding(tokens)

        return x + self.position_embedding


class CLIPLayer(nn.Module):
    def __init__(self, n_heads: int, n_embed: int):
        super().__init__()

        self.layernorm1 = nn.LayerNorm(n_embed)
        self.attention = SelfAttention(n_heads, n_embed)
        self.layernorm2 = nn.LayerNorm(n_embed)

        self.linear1 = nn.Linear(n_embed, n_embed * 4)
        self.linear2 = nn.Linear(n_embed * 4, n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue = x

        x = self.layernorm1(x)
        x = self.attention(x)

        x = residue + x

        residue = x

        x = self.layernorm2(x)
        x = self.linear1(x)

        x = x * torch.sigmoid(1.702 * x)  # QuickGELU

        x = self.linear2(x)

        return residue + x


class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for _ in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.Tensor:
        tokens = tokens.type(torch.long)

        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        return self.layernorm(state)
