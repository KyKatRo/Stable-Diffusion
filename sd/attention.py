import torch
from torch import nn
from torch.nn import functional as F
import math


class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def split_heads(self, x, batch_size, sequence_length):
        """Reshape and transpose the input tensor for multi-head attention."""
        return x.view(batch_size, sequence_length, self.n_heads, self.d_head).transpose(1, 2)

    def forward(self, x, causal_mask=False):
        """Compute multi-head self-attention.

        :param x:
        :param causal_mask:
        :return:
        """
        batch_size, sequence_length, _ = x.shape
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        q = self.split_heads(q, batch_size, sequence_length)
        k = self.split_heads(k, batch_size, sequence_length)
        v = self.split_heads(v, batch_size, sequence_length)

        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight = weight.masked_fill(mask, float('-inf'))

        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        output = weight @ v
        output = output.transpose(1, 2).reshape(batch_size, sequence_length, -1)
        output = self.out_proj(output)

        return output


class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def reshape_for_attention(self, x, batch_size):
        """
        Reshapes input tensor x for multi-head attention processing.
        :param x: Input tensor to reshape
        :param batch_size: Batch size of input tensor
        :return: Reshaped tensor
        """
        return x.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)

    def forward(self, x, y):
        batch_size, sequence_length, _ = x.shape

        # Project the inputs to the query, key, and value spaces
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        # Reshape and transpose for multi-head attention
        q = self.reshape_for_attention(q, batch_size)
        k = self.reshape_for_attention(k, batch_size)
        v = self.reshape_for_attention(v, batch_size)

        # Calculate the attention scores
        weight = q @ k.transpose(-1, -2)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        # Apply the attention scores to the value vectors
        output = weight @ v

        # Reshape the output and apply the final linear layer
        output = output.transpose(1, 2).contiguous().view(batch_size, sequence_length, -1)
        output = self.out_proj(output)

        return output
