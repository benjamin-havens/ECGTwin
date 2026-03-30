"""DiT backbone variant using explicit cross-attention conditioning."""

# --------------------------------------------------------
# References:
# DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
# --------------------------------------------------------

import torch
import torch.nn as nn
from ecgtwin.models.attention import CrossAttention
from ecgtwin.models.embedder import RoPEEmbedder, TimestepEmbedder


class DiTBlock_ATTN(nn.Module):
    """
    A DiT block with cross attention conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn1 = nn.MultiheadAttention(hidden_size, num_heads=num_heads, bias=True, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.cnorm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn2 = CrossAttention(hidden_size, num_heads=num_heads, bias=True)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.norm3 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = nn.Sequential(
                                 nn.Linear(hidden_size, mlp_hidden_dim),
                                 nn.GELU(approximate="tanh"),
                                 nn.Linear(mlp_hidden_dim, hidden_size)
        )

    def forward(self, x, c, mask):
        # (B, L, D)
        x1 = self.norm1(x)
        x1 = self.attn1(x1, x1, x1)[0]
        x = x + x1

        x2 = self.norm2(x)
        # c: (B, L_e + L_p, dim)
        c = self.cnorm(c)
        x2 = self.attn2(x2, c, mask)
        x = x + x2

        x3 = self.norm3(x)
        x3 = self.mlp(x3)
        x = x + x3
        # (B, L, D)
        return x


class ECG_DiT_ATTN(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        in_channels=4,
        hidden_size=256,
        text_embed_dim=768,
        pat_info_length=3,
        base_vector_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels 
        self.num_heads = num_heads

        self.x_embedder = nn.Linear(in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        self.text_projector = nn.Linear(text_embed_dim + pat_info_length, hidden_size)

        self.ib_projector = nn.Linear(base_vector_dim, hidden_size)

        # Will use fixed sin-cos embedding:
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.pos_embed = RoPEEmbedder(dim=hidden_size)

        self.blocks = nn.ModuleList([
            DiTBlock_ATTN(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        self.final_layer = nn.Sequential(
        nn.LayerNorm(hidden_size),
        nn.Linear(hidden_size, self.out_channels)
        )
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer[1].weight, 0)
        nn.init.constant_(self.final_layer[1].bias, 0)

    def forward(self, x, t, text_embed, text_embed_mask, p, base_vector):
        """
        Forward pass of DiT.
        x: (B, C, L) tensor of spatial inputs (signals or latent representations of signals)
        t: (B,) tensor of diffusion timesteps
        text_embed: (B, L, dim) text embedding sequence of each report
        text_embed_mask: (B, L) mask of valid text embedding
        p: (B, L, Dim) patient info vector
        base vector: (B, dim) extracted individual base vector

        return (B, C, L)
        """
        x = x.transpose(2, 1)
        x = self.x_embedder(x)                   
        x = self.pos_embed(x)                    # (B, L, D)

        t = self.t_embedder(t).unsqueeze(1)      # (B, 1, D)
        b = self.ib_projector(base_vector).unsqueeze(1) # (B, 1, D)
        x = x + t + b

        p = p.unsqueeze(1).repeat(1, text_embed.shape[1], 1)# (B, L, D)
        c = torch.concat([text_embed, p], dim=-1)
        c = self.text_projector(c)      # (B, L, D)

        for block in self.blocks:
            x = block(x, c, text_embed_mask)  # (B, L, D)
        x = self.final_layer(x)                
        x = x.transpose(2, 1)
        return x

if __name__ == '__main__':
    """
    Test
    """
    DiT = ECG_DiT_ATTN(depth=7)
    batch_size = 64 
    vae_latent = torch.randn((batch_size, 4, 128))
    t = torch.randperm(1000)[:batch_size]
    text_embed = torch.randn((batch_size, 17, 768))
    text_embed_mask = torch.ones((batch_size, 17))
    pat_info = torch.randn((batch_size, 3))
    base_vector = torch.randn((batch_size, 256))
    
    output = DiT(vae_latent, t, text_embed, text_embed_mask, pat_info, base_vector)
    print(output.shape)
    total_params = sum(p.numel() for p in DiT.parameters())
    print(f'Total number of parameters in the model: {total_params}')
