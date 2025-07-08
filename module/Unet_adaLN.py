import torch
import torch.nn as nn
import torch.nn.functional as F

from module.Embedder import TimestepEmbedder, PatientInfoEmbedder
    
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class Block(nn.Module):
    def __init__(self, n_inputs, n_outputs, 
                 kernel_size, n_heads, hidden_dim, text_embed_dim, patient_info_length, base_vector_dim):
        super(Block, self).__init__()
        self.pre_shortcut_convs = nn.Conv1d(n_inputs, hidden_dim, kernel_size, padding="same")# padding="same"
        self.shortcut_convs = nn.Conv1d(hidden_dim, hidden_dim, 1, padding="same")#padding="same"
        self.post_shortcut_convs = nn.Conv1d(hidden_dim, n_outputs, kernel_size, padding="same")#, padding="same"
        self.res_conv = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else nn.Identity()

        self.mlp = nn.Sequential(
                                 nn.Linear(hidden_dim, 4 * hidden_dim),
                                 nn.GELU(approximate="tanh"),
                                 nn.Linear(4 * hidden_dim, hidden_dim)
        )

        self.norm1 = nn.GroupNorm(1, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attention_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6) 

        self.p_embedder = PatientInfoEmbedder(patient_info_length, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=n_heads, bias=True, batch_first=True)
        self.text_projector = nn.Linear(text_embed_dim, hidden_dim)
        self.ib_projector = nn.Linear(base_vector_dim, hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )

        self.time_emb = TimestepEmbedder(hidden_dim)

    def forward(self, x, t, text_embed, text_embed_mask, pat_info, base_vector):
        # x: (B, C_in, L)
        initial_x = x
        # t: (B, C_short)
        t = self.time_emb(t)
        if text_embed_mask is None:
            y = torch.mean(text_embed, dim=1)
        else:
            y = torch.sum(text_embed, dim=1) / torch.sum(text_embed_mask, dim=1, keepdim=True)
        # y: (B, L, dim) -> (B, C_short)
        y = self.text_projector(y)               
        # pat_info: (B, 3) -> (B, C_short)
        p = self.p_embedder(pat_info).sum(dim=1)
        # (B, C_short)
        b = self.ib_projector(base_vector)
        c = t + y + p + b

        # shortcut: (B, C_short, L)
        shortcut = self.pre_shortcut_convs(x)
        shortcut = self.norm1(shortcut)
        shortcut = F.mish(shortcut)
        shortcut = self.shortcut_convs(shortcut)

        # shortcut: (B, L, C_short)
        shortcut = shortcut.transpose(-1, -2)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        attn_input = modulate(self.attention_norm(shortcut), shift_msa, scale_msa)
        attn_out = self.attn(attn_input, attn_input, attn_input) 
        shortcut = shortcut + gate_msa.unsqueeze(1) * attn_out[0]

        mlp_input = modulate(self.norm2(shortcut), shift_mlp, scale_mlp)
        out = shortcut + gate_mlp.unsqueeze(1) * self.mlp(mlp_input)

        # out: (B, C_out, L)
        out = out.transpose(-1, -2)
        out = self.post_shortcut_convs(out)
        out = F.mish(out)
        out = (out + self.res_conv(initial_x))# / math.sqrt(2.0)
        return out


# modified to 2D
class DownsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs,  
                 kernel_size, n_heads, hidden_dim, text_embed_dim, patient_info_length, base_vector_dim):
        super(DownsamplingBlock, self).__init__()
        self.down = nn.Conv1d(in_channels=n_outputs, out_channels=n_outputs, kernel_size=3, stride=2, padding=1)
        self.block = Block(n_inputs, n_outputs,  kernel_size=kernel_size, n_heads=n_heads, hidden_dim=hidden_dim, text_embed_dim=text_embed_dim, patient_info_length=patient_info_length, base_vector_dim=base_vector_dim)

    def forward(self, x, t, text_embed, text_embed_mask, patient_info, base_vector):
        h = self.block(x, t, text_embed, text_embed_mask, patient_info, base_vector)
        # DOWNSAMPLING
        out = self.down(h)
        return h, out


class UpsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs,  
                 kernel_size, up_dim, n_heads, hidden_dim, text_embed_dim, patient_info_length, base_vector_dim):
        super(UpsamplingBlock, self).__init__()
        self.block = Block(n_inputs, n_outputs, kernel_size=kernel_size, n_heads=n_heads, hidden_dim=hidden_dim, text_embed_dim=text_embed_dim, patient_info_length=patient_info_length, base_vector_dim=base_vector_dim)

        if up_dim is None:
            self.up = nn.ConvTranspose1d(n_inputs // 2, n_inputs // 2, kernel_size=4, stride=2, padding=1)#padding=1
        else:
            self.up = nn.ConvTranspose1d(up_dim, up_dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x, h, t, text_embed, text_embed_mask, pat_info, base_vector):
        x = self.up(x) 
        if h is not None:
            x = torch.cat([x, h], dim=1)
        out = self.block(x, t, text_embed, text_embed_mask, pat_info, base_vector)
        return out


class BottleneckNet(nn.Module):
    def __init__(self, n_channels, 
                 kernel_size, n_heads, hidden_dim, text_embed_dim, patient_info_length, base_vecotr_dim):
        super(BottleneckNet, self).__init__()
        self.time_emb = TimestepEmbedder(n_channels)        
        self.bottleneck_conv1 = nn.Conv1d(n_channels, n_channels , kernel_size=kernel_size, padding="same")
        self.bottleneck_conv2 = nn.Conv1d(n_channels, n_channels , kernel_size=kernel_size, padding="same")

        self.bottleneck_norm1 = nn.GroupNorm(1, n_channels)
        self.bottleneck_norm2 = nn.LayerNorm(n_channels, elementwise_affine=False, eps=1e-6)
        self.attention_norm = nn.LayerNorm(n_channels, elementwise_affine=False, eps=1e-6) 

        self.text_projector = nn.Linear(text_embed_dim, n_channels)
        self.p_embedder = PatientInfoEmbedder(patient_info_length, n_channels)
        self.ib_projector = nn.Linear(base_vecotr_dim, n_channels)
        self.attn = nn.MultiheadAttention(n_channels, num_heads=n_heads, bias=True, batch_first=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(n_channels, 6 * n_channels, bias=True)
        )

        self.mlp = nn.Sequential(
                                 nn.Linear(n_channels, hidden_dim),
                                 nn.GELU(approximate="tanh"),
                                 nn.Linear(hidden_dim, n_channels)
        )

    def forward(self, x, t, text_embed, text_embed_mask, pat_info, base_vector):
        out = x

        out = self.bottleneck_conv1(out)
        out = self.bottleneck_norm1(out)
        out = F.mish(out)
        out = self.bottleneck_conv2(out)

        t = self.time_emb(t)
        if text_embed_mask is None:
            y = torch.mean(text_embed, dim=1)
        else:
            y = torch.sum(text_embed, dim=1) / torch.sum(text_embed_mask, dim=1, keepdim=True)
        # y: (B, L, dim) -> (B, C_short)
        y = self.text_projector(y)               
        # pat_info: (B, 3) -> (B, C_short)
        p = self.p_embedder(pat_info).sum(dim=1)
        # base vector: (B, L, dim) -> (B, C_short)
        b = self.ib_projector(base_vector)
        c = t + y + p + b

        out = out.transpose(-1, -2)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        attn_input = modulate(self.attention_norm(out), shift_msa, scale_msa)
        attn_out = self.attn(attn_input, attn_input, attn_input) 
        out = out + gate_msa.unsqueeze(1) * attn_out[0]

        mlp_input = modulate(self.bottleneck_norm2(out), shift_mlp, scale_mlp)
        out = out + gate_mlp.unsqueeze(1) * self.mlp(mlp_input)
        out = out.transpose(-1, -2)

        out = F.mish(out)
        out = (x + out) #/ math.sqrt(2)
        return out


class ECG_UNET_adaLN(nn.Module):
    def __init__(self, kernel_size=7, num_levels=7, n_heads=8, n_channels=4, text_embed_dim=768, patient_info_length=3, base_vector_dim=256):
        super(ECG_UNET_adaLN, self).__init__()

        self.num_levels = num_levels
        input_channels_list = []
        output_channels_list = []

        for i in range(num_levels - 1):
            input_channels_list.append(n_channels * 2**i)  
        for i in range(num_levels - 1):    
            x = 2 * input_channels_list[num_levels - i - 2]
            input_channels_list.append(x)
                
        for i in range(num_levels - 1):
            output_channels_list.append(2 * input_channels_list[i]) 
        for i in range(num_levels - 1):    
            x = output_channels_list[num_levels - i - 2] // 2 
            output_channels_list.append(x)

        for i in range(num_levels - 2):
            k = 2 * (num_levels - 1) - i - 1
            input_channels_list[k] += output_channels_list[i]

        n_hidden_state_list = [channel * 2 for channel in input_channels_list]
        # print(input_channels_list)
        # print(n_hidden_state_list)

        # Only odd filter kernels allowed
        assert(kernel_size % 2 == 1)
        self.downsampling_blocks = nn.ModuleList()
        self.upsampling_blocks = nn.ModuleList()

        for i in range(self.num_levels - 1):
            self.downsampling_blocks.append(
                DownsamplingBlock(n_inputs=input_channels_list[i], n_outputs=output_channels_list[i],
                            kernel_size=kernel_size, n_heads=n_heads, 
                            hidden_dim=n_hidden_state_list[i], text_embed_dim=text_embed_dim, 
                            patient_info_length=patient_info_length, base_vector_dim=base_vector_dim))

        self.bottelneck = BottleneckNet(n_channels=input_channels_list[num_levels],
                                        kernel_size=kernel_size, n_heads=4, hidden_dim=32, 
                                        text_embed_dim=text_embed_dim, patient_info_length=patient_info_length, 
                                        base_vecotr_dim=base_vector_dim)

        i = self.num_levels - 1
        self.upsampling_blocks.append(
            UpsamplingBlock(n_inputs=input_channels_list[i], n_outputs=output_channels_list[i],
                                  kernel_size=kernel_size, up_dim=input_channels_list[i],
                                  n_heads=n_heads, hidden_dim=n_hidden_state_list[i], text_embed_dim=text_embed_dim, 
                                  patient_info_length=patient_info_length, base_vector_dim=base_vector_dim))
        for i in range(self.num_levels, 2*self.num_levels - 2):
            self.upsampling_blocks.append(
                UpsamplingBlock(n_inputs=input_channels_list[i], n_outputs=output_channels_list[i],
                                  kernel_size=kernel_size, up_dim=None, n_heads=n_heads, hidden_dim=n_hidden_state_list[i],
                                  text_embed_dim=text_embed_dim, patient_info_length=patient_info_length, base_vector_dim=base_vector_dim))


        self.output_conv = nn.Sequential(nn.Conv1d(output_channels_list[-1], n_channels, 3, padding="same"), nn.Mish(),
                                         nn.Conv1d(n_channels, n_channels, 1, padding="same"))

    def forward(self, x, t, text_embed, text_embed_mask, pat_info, base_vector):
        '''
        '''
        shortcuts = []
        out = x

        # DOWNSAMPLING BLOCKS
        for block in self.downsampling_blocks:
            h, out = block(out, t, text_embed, text_embed_mask, pat_info, base_vector)
            shortcuts.append(h)
            # print(out.shape)
        del shortcuts[-1]
        #out = self.downsampling_blocks[-1](out)

        # BOTTLENECK CONVOLUTION
        out = self.bottelneck(out, t, text_embed, text_embed_mask, pat_info, base_vector) 
        # print(out.shape)      

        # UPSAMPLING BLOCKS
        out = self.upsampling_blocks[0](out, None, t, text_embed, text_embed_mask, pat_info, base_vector)
        # print(out.shape)
        for idx, block in enumerate(self.upsampling_blocks[1:]):
            out = block(out, shortcuts[-1-idx], t, text_embed, text_embed_mask, pat_info, base_vector)
            # print(out.shape)

        # OUTPUT CONV
        out = self.output_conv(out)
        return out


if __name__ == '__main__':
    """
    Test
    """
    unet = ECG_UNET_adaLN(num_levels=7)
    batch_size = 64 
    vae_latent = torch.randn((batch_size, 4, 128))
    t = torch.randperm(1000)[:batch_size]
    text_embed = torch.randn((batch_size, 17, 768))
    text_embed_mask = torch.ones((batch_size, 17))
    pat_info = torch.randn((batch_size, 3))
    base_vector = torch.randn((batch_size, 256))
    
    output = unet(vae_latent, t, text_embed, text_embed_mask, pat_info, base_vector)
    print(output.shape)
    total_params = sum(p.numel() for p in unet.parameters())
    print(f'Total number of parameters in the model: {total_params}')
