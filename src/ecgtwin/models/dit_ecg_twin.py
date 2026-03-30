"""Primary ECGTwin DiT backbone with base-vector conditioning."""

# --------------------------------------------------------
# References:
# DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
# --------------------------------------------------------

import torch
import torch.nn as nn
from ecgtwin.models.attention import CrossAttention
from ecgtwin.models.embedder import PositionalEmbedder, RoPEEmbedder, TimestepEmbedder


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock_ECGTwin(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn1 = nn.MultiheadAttention(hidden_size, num_heads=num_heads, bias=True, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn2 = CrossAttention(hidden_size, num_heads=num_heads, bias=True)
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

    def forward(self, x, c, c2, mask, need_weights=False, average_attn_weights=True):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        attn_input = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out = self.attn1(attn_input, attn_input, attn_input) 
        x = x + gate_msa.unsqueeze(1) * attn_out[0]

        c2 = self.cnorm(c2)
        if need_weights:
            attn_out2, attn_map = self.attn2(self.norm2(x), c2, mask, need_weights=True, average_attn_weights=average_attn_weights)
            x = x + attn_out2
        else:
            x = x + self.attn2(self.norm2(x), c2, mask, need_weights=False)

        mlp_input = modulate(self.norm3(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_input)

        if need_weights:
            return x, attn_map
        else:
            return x

    def inject_attn(self, x, c, c2, attn_map):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        attn_input = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out = self.attn1(attn_input, attn_input, attn_input) 
        x = x + gate_msa.unsqueeze(1) * attn_out[0]

        c2 = self.cnorm(c2)
        x = x + self.attn2.forward_with_given_attn_map(self.norm2(x), c2, attn_map)

        mlp_input = modulate(self.norm3(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_input)
        return x


class DiT_ECGTwin(nn.Module):
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

        self.pos_embed = RoPEEmbedder(dim=hidden_size)
        self.c_pos_embed = PositionalEmbedder(embed_size=hidden_size, max_len=25)

        self.blocks = nn.ModuleList([
            DiTBlock_ECGTwin(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
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

    def forward(self, x, t, text_embed, text_embed_mask, p, base_vector, need_weights=False):
        """
        Forward pass of DiT.
        x: (B, C, L) tensor of spatial inputs (signals or latent representations of signals)
        t: (B,) tensor of diffusion timesteps
        text_embed: (B, L, dim) text embedding sequence of each report
        text_embed_mask: (B, L) mask of valid text embedding
        p: (B, D) patient info vector
        base vector: (B, dim) extracted individual base vector

        return (B, C, L)
        """
        x = x.transpose(2, 1)
        x = self.x_embedder(x)   
        x = self.pos_embed(x)                    # (B, L, D)

        t = self.t_embedder(t)                   # (B, D)
        b = self.ib_projector(base_vector)       # (B, D)
        c = t + b                                # (B, D)

        p = p.unsqueeze(1).repeat(1, text_embed.shape[1], 1)# (B, L, D)
        c2 = torch.concat([p, text_embed], dim=-1)         # (B, L, D)

        c2 = self.text_projector(c2)      # (B, L, D)
        c2 = self.c_pos_embed(c2)

        if need_weights:
            attn_map_list = []
            for block in self.blocks:
                x, attn_map = block(x, c, c2, text_embed_mask, need_weights=True)      # (B, L, D)
                attn_map_list.append(attn_map)
        else:
            for block in self.blocks:
                x = block(x, c, c2, text_embed_mask, need_weights=False)            # (B, L, D)

        x = self.final_layer(x)                
        x = x.transpose(2, 1)

        if need_weights:
            return x, attn_map_list
        else:
            return x

class ECGTwin_Edit(DiT_ECGTwin):

    def _input_process(self, x):
        x = x.transpose(2, 1)
        x = self.x_embedder(x)   
        x = self.pos_embed(x)                    # (B, L, D)
        return x
    
    def _output_process(self, x):
        x = self.final_layer(x)                
        x = x.transpose(2, 1)
        return x

    @torch.no_grad
    def swap_report(self, x_src, x_edit, t, text_embed_src, text_embed_edit, pat_info, base_vector):
        """
        Swap new report tokens.
        Using attn map from previous report and perform cross attention with new reports
        """

        x_src = self._input_process(x_src)
        x_edit = self._input_process(x_edit)

        t = self.t_embedder(t)                   # (B, D)
        b = self.ib_projector(base_vector)       # (B, D)
        c = t + b                                # (B, D)

        p = pat_info.unsqueeze(1).repeat(1, text_embed_src.shape[1], 1)# (B, L, D)
        c2 = torch.concat([p, text_embed_src], dim=-1)         # (B, L, D)
        c2 = self.text_projector(c2)      # (B, L, D)
        c2 = self.c_pos_embed(c2)

        p = pat_info.unsqueeze(1).repeat(1, text_embed_edit.shape[1], 1)# (B, L, D)
        c3 = torch.concat([p, text_embed_edit], dim=-1)
        c3 = self.text_projector(c3)      # (B, L, D)
        c3 = self.c_pos_embed(c3)

        for block in self.blocks:
            x_src, attn_map = block(x_src, c, c2, None, True, average_attn_weights=False)            # (B, L, D)
            x_edit = block.inject_attn(x_edit, c, c3, attn_map)
        
        x_src = self._output_process(x_src)
        x_edit = self._output_process(x_edit)

        return x_src, x_edit

    @torch.no_grad
    def add_report(self, x_src, x_edit, t, text_embed_src, text_embed_add, pat_info, base_vector, weight=1.0):
        """
        Add new report tokens.
        Using attn map from previous report and add new column based on new reports
        """

        x_src = self._input_process(x_src)
        x_edit = self._input_process(x_edit)

        t = self.t_embedder(t)                   # (B, D)
        b = self.ib_projector(base_vector)       # (B, D)
        c = t + b                                # (B, D)

        p = pat_info.unsqueeze(1).repeat(1, text_embed_src.shape[1], 1)# (B, L, D)
        c2 = torch.concat([p, text_embed_src], dim=-1)         # (B, L, D)
        c2 = self.text_projector(c2)      # (B, L, D)
        c2 = self.c_pos_embed(c2)

        text_embed_edit = torch.concat([text_embed_src, text_embed_add], dim=1)
        p = pat_info.unsqueeze(1).repeat(1, text_embed_edit.shape[1], 1)# (B, L, D)
        c3 = torch.concat([p, text_embed_edit], dim=-1)
        c3 = self.text_projector(c3)      # (B, L, D)
        c3 = self.c_pos_embed(c3)

        for block in self.blocks:
            x_src, attn_map = block(x_src, c, c2, None, True, average_attn_weights=False)
            _, attn_map_add = block(x_edit, c, c3, None, True, average_attn_weights=False)

            # (B, num_heads, L_q, L_k)
            new_attn_map = torch.concat([attn_map, weight * attn_map_add[:, :, :, -1:]], dim=-1)
            x_edit = block.inject_attn(x_edit, c, c3, new_attn_map)
        
        x_src = self._output_process(x_src)
        x_edit = self._output_process(x_edit)

        return x_src, x_edit


if __name__ == '__main__':
    """
    Test
    """
    DiT = DiT_ECGTwin(depth=7)
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
