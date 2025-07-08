import torch
import torch.nn as nn
import torch.nn.functional as F

from module.Attention import CrossAttention
from module.Embedder import TimestepEmbedder, PositionalEmbedder
    

class Block(nn.Module):
    def __init__(self, n_inputs, n_outputs, 
                 kernel_size, n_heads, hidden_dim, text_embed_dim, patient_info_length, base_vector_dim):
        super(Block, self).__init__()
        self.pre_shortcut_convs = nn.Conv1d(n_inputs, hidden_dim, kernel_size, padding="same")# padding="same"
        self.shortcut_convs = nn.Conv1d(hidden_dim, hidden_dim, 1, padding="same")#padding="same"
        self.post_shortcut_convs = nn.Conv1d(hidden_dim, n_outputs, kernel_size, padding="same")#, padding="same"

        self.layer_norm1 = nn.GroupNorm(1, hidden_dim)
        self.layer_norm2 = nn.GroupNorm(1, hidden_dim)
        self.cnorm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.res_conv = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else nn.Identity()
        # self.text_embedding_layer = nn.Linear(text_embed_dim, dim)

        self.cross_attention = CrossAttention(hidden_size=hidden_dim, num_heads=n_heads, bias=True)

        self.time_emb = TimestepEmbedder(n_inputs)
        self.c_pos_emb = PositionalEmbedder(hidden_dim)
        self.ib_projector = nn.Linear(base_vector_dim, n_inputs)
        self.text_projector = nn.Linear(text_embed_dim + patient_info_length, hidden_dim)

    def forward(self, x, t, text_embed, text_embed_mask, pat_info, base_vector):
        # x: (B, C_in, L)
        initial_x = x
        # t: (B, C_in, 1)
        t = self.time_emb(t).unsqueeze(-1)
        # base vector: (B, L, dim) -> (B, C_short, 1)
        b = self.ib_projector(base_vector).unsqueeze(-1)
        x = x + t + b

        # shortcut: (B, C_short, L)
        shortcut = self.pre_shortcut_convs(x)
        shortcut = self.layer_norm1(shortcut)
        shortcut = F.mish(shortcut)
        shortcut = self.shortcut_convs(shortcut)

        # shortcut: (B, L, C_short)
        p = pat_info.unsqueeze(1).repeat(1, text_embed.shape[1], 1)# (B, L, D)
        c2 = torch.concat([text_embed, p], dim=-1)
        c2 = self.text_projector(c2)
        c2 = self.c_pos_emb(c2)
        c2 = self.cnorm(c2)
        shortcut = shortcut.transpose(-1, -2)
        shortcut = shortcut + self.cross_attention(shortcut, c2, text_embed_mask)
        shortcut = shortcut.transpose(-1, -2)

        # out: (B, C_out, L)
        shortcut = self.layer_norm2(shortcut)
        out = self.post_shortcut_convs(shortcut)
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
                 kernel_size, n_heads, text_embed_dim, patient_info_length, base_vector_dim):
        super(BottleneckNet, self).__init__()
        self.time_emb = TimestepEmbedder(n_channels)        
        self.c_pos_emb = PositionalEmbedder(n_channels)
        self.bottleneck_conv1 = nn.Conv1d(n_channels, n_channels , kernel_size=kernel_size, padding="same")
        self.bottleneck_conv1_2 = nn.Conv1d(n_channels, n_channels , kernel_size=kernel_size, padding="same")
        self.bottleneck_conv2 = nn.Conv1d(n_channels, n_channels, kernel_size=kernel_size, padding="same")
        self.bottleneck_layer_norm1 = nn.GroupNorm(1, n_channels)
        self.bottleneck_layer_norm2 = nn.GroupNorm(1, n_channels)
        self.cnorm = nn.LayerNorm(n_channels, elementwise_affine=False, eps=1e-6)

        self.cross_attention = CrossAttention(num_heads=n_heads, hidden_size=n_channels, bias=True)
        self.ib_projector = nn.Linear(base_vector_dim, n_channels)
        self.text_projector = nn.Linear(text_embed_dim + patient_info_length, n_channels)

    def forward(self, x, t, text_embed, text_embed_mask, pat_info, base_vector):
        # (B, C, L)
        out = x
        tt = self.time_emb(t).unsqueeze(-1)
        b = self.ib_projector(base_vector).unsqueeze(-1)
        out = out + tt + b

        out = self.bottleneck_conv1(out)
        out = self.bottleneck_layer_norm1(out)
        out = F.mish(out)
        out = self.bottleneck_conv1_2(out)

        p = pat_info.unsqueeze(1).repeat(1, text_embed.shape[1], 1)# (B, L, D)
        c2 = torch.concat([text_embed, p], dim=-1)
        c2 = self.text_projector(c2)
        c2 = self.c_pos_emb(c2)
        c2 = self.cnorm(c2)
        out = out.transpose(-1, -2)
        out = out + self.cross_attention(out, c2, text_embed_mask)
        out = out.transpose(-1, -2)

        out = self.bottleneck_layer_norm2(out)
        out = self.bottleneck_conv2(out)
        out = F.mish(out)
        
        out = (x + out) #/ math.sqrt(2)
        return out


class ECG_UNET_ATTN(nn.Module):
    def __init__(self, kernel_size=7, num_levels=7, n_heads=8, n_channels=4, text_embed_dim=768, patient_info_length=3, base_vector_dim=256):
        super(ECG_UNET_ATTN, self).__init__()

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
                                        kernel_size=kernel_size, n_heads=n_heads, 
                                        text_embed_dim=text_embed_dim, patient_info_length=patient_info_length,
                                        base_vector_dim=base_vector_dim)

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
    unet = ECG_UNET_ATTN(num_levels=7)
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