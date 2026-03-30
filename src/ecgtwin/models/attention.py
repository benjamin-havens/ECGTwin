"""Attention primitives shared by multiple ECGTwin model families."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):

    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask=False):
        # x: (Batch_size, Seq_len, Dim)
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # (B, S, D) -> (B, S, D * 3) -> 3 * (B, S, D) NOTE: nn.Linear multiplys last dimension of any given vector
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        
        # (B, S, D) -> (B, S, H, D/H) -> (B, H, S, D/H)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # (B, H, S, S)
        weight = q @ k.transpose(-1, -2)
        
        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
        
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        # (B, H, S, S) @ (B, H, S, D/H) -> (B, H, S, D/H)
        output = weight @ v

        # (B, H, S, D/H) -> (B, S, H, D/H)
        output = output.transpose(1, 2)

        output = output.reshape(input_shape)

        output = self.out_proj(output)

        # (B, S, D)
        return output


class CrossAttention(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int, bias=True):
        super().__init__()

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.kv_proj = nn.Linear(hidden_size, 2 * hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.q_norm = nn.LayerNorm(hidden_size)
        self.k_norm = nn.LayerNorm(hidden_size)
        self.v_norm = nn.LayerNorm(hidden_size)
        self.num_heads = num_heads
        self.d_head = hidden_size // num_heads

    def forward(self, x: torch.Tensor, c: torch.Tensor, mask: torch.Tensor=None, need_weights: bool=False, average_attn_weights: bool=True):
        # x: (Batch_size, Seq_len, Dim)
        # c: (B, Lk, D)
        # mask: (B, Lk)
        interim_shape_q = (x.shape[0], x.shape[1], self.num_heads, self.d_head)
        interim_shape_kv = (c.shape[0], c.shape[1], self.num_heads, self.d_head)

        q = self.q_proj(x)
        q = self.q_norm(q)

        # (B, L, D) -> (B, L, D * 2) -> 2 * (B, L, D)
        k, v = self.kv_proj(c).chunk(2, dim=-1)
        k = self.k_norm(k)
        v = self.v_norm(v)
        
        # (B, L, D) -> (B, L, H, D/H) -> (B, H, L, D/H)
        q = q.view(interim_shape_q).transpose(1, 2)
        k = k.view(interim_shape_kv).transpose(1, 2)
        v = v.view(interim_shape_kv).transpose(1, 2)

        # (B, H, Lq, Lk)
        weight = q @ k.transpose(-1, -2)
        
        if mask is not None:
            # Expand mask to (B, 1, 1, Lk) for broadcasting
            mask = mask.unsqueeze(1).unsqueeze(2)
            weight = weight.masked_fill(mask == 0, float('-inf'))

        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        # (B, H, Lq, Lk) @ (B, H, Lk, D/H) -> (B, H, Lq, D/H)
        output = weight @ v

        # (B, H, Lq, D/H) -> (B, Lq, H, D/H)
        output = output.transpose(1, 2)
        output = output.reshape(x.shape[0], x.shape[1], x.shape[2])
        output = self.out_proj(output)

        # (B, L, D)
        if need_weights:
            if average_attn_weights:
                weight = weight.mean(dim=1)
            return output, weight
        else:
            return output

    def forward_with_given_attn_map(self, x: torch.Tensor, c: torch.Tensor, attn_map_tar: torch.Tensor):
        # x: (Batch_size, Seq_len, Dim)
        # c: (B, Lk, D)
        # mask: (B, Lk)
        interim_shape_kv = (c.shape[0], c.shape[1], self.num_heads, self.d_head)

        # (B, L, D) -> (B, L, D * 2) -> 2 * (B, L, D)
        _, v = self.kv_proj(c).chunk(2, dim=-1)
        v = self.v_norm(v)
        
        # (B, L, D) -> (B, L, H, D/H) -> (B, H, L, D/H)
        v = v.view(interim_shape_kv).transpose(1, 2)

        # (B, H, Lq, Lk) @ (B, H, Lk, D/H) -> (B, H, Lq, D/H)
        output = attn_map_tar @ v

        # (B, H, Lq, D/H) -> (B, Lq, H, D/H)
        output = output.transpose(1, 2)
        output = output.reshape(x.shape[0], x.shape[1], x.shape[2])
        output = self.out_proj(output)

        # (B, L, D)
        return output
