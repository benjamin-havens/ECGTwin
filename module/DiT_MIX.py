# --------------------------------------------------------
# References:
# DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
# --------------------------------------------------------

import torch
import torch.nn as nn

from module.Embedder import TimestepEmbedder, RoPEEmbedder, PatientInfoEmbedder, PositionalEmbedder


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock_MIX(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn1 = nn.MultiheadAttention(hidden_size, num_heads=num_heads, bias=True, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn2 = nn.MultiheadAttention(hidden_size, num_heads=num_heads, bias=True, batch_first=True)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cnorm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
                                 nn.Linear(hidden_size, mlp_hidden_dim),
                                 nn.GELU(approximate="tanh"),
                                 nn.Linear(mlp_hidden_dim, hidden_size)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, c2, mask):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        attn_input = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out = self.attn1(attn_input, attn_input, attn_input) 
        x = x + gate_msa.unsqueeze(1) * attn_out[0]

        c2 = self.cnorm(c2)
        x = x + self.attn2(self.norm2(x), c2, c2, mask)[0]

        mlp_input = modulate(self.norm3(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_input)
        return x

class ECG_DiT_MIX(nn.Module):
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
        
        self.p_embedder = PatientInfoEmbedder(pat_info_length=pat_info_length, dim=hidden_size)
        self.text_projector = nn.Linear(text_embed_dim, hidden_size)

        self.ib_projector = nn.Linear(base_vector_dim, hidden_size)

        # Will use fixed sin-cos embedding:
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.pos_embed = RoPEEmbedder(dim=hidden_size)

        self.blocks = nn.ModuleList([
            DiTBlock_MIX(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = nn.Linear(hidden_size, self.out_channels)
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

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        # nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        # nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        # nn.init.constant_(self.final_layer.linear.weight, 0)
        # nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, text_embed, text_embed_mask, p, base_vector):
        """
        Forward pass of DiT.
        x: (B, C, L) tensor of spatial inputs (signals or latent representations of signals)
        t: (B,) tensor of diffusion timesteps
        text_embed: (B, L, dim) text embedding sequence of each report
        text_embed_mask: (B, L) mask of valid text embedding
        p: (B, L) patient info vector
        base vector: (B, dim) extracted individual base vector

        return (B, C, L)
        """
        x = x.transpose(2, 1)
        x = self.x_embedder(x)   
        x = self.pos_embed(x)                    # (B, L, D)

        t = self.t_embedder(t)                   # (B, D)
        b = self.ib_projector(base_vector)       # (B, D)
        c = t + b                                # (B, D)

        y = self.text_projector(text_embed)      # (B, L, D)
        p = self.p_embedder(p)                   # (B, L, D)

        c2 = torch.concat([p, y], dim=1)         # (B, L, D)
        if text_embed_mask is not None:
            mask = torch.concat([torch.ones(p.shape[0], p.shape[1], device=text_embed_mask.device), text_embed_mask], dim=1)
        else:
            mask = None

        for block in self.blocks:
            x = block(x, c, c2, mask)            # (B, L, D)

        # for block in self.blocks:
        #     x = block(x, c, y, text_embed_mask)

        x = self.final_layer(x)                
        x = x.transpose(2, 1)
        return x

if __name__ == '__main__':
    """
    Test
    """
    DiT = ECG_DiT_MIX(depth=7)
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
