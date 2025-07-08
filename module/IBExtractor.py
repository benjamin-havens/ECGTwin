import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from module.Embedder import RoPEEmbedder, PatientInfoEmbedder

# Feed Forward Network
class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_hidden_size):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, ff_hidden_size)
        self.fc2 = nn.Linear(ff_hidden_size, embed_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class IBEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_size, dropout):
        super(IBEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.feed_forward = FeedForward(embed_dim, ff_hidden_size)

        self.condition_norm = nn.LayerNorm(embed_dim)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.layer_norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, c, mask):
        # x, text_embed, m: (B, L, dim)
        # p: (B, L_p, 1)

        # Self-attention sublayer with residual connection
        x1 = self.layer_norm1(x)
        attn_output = self.self_attn(x1, x1, x1, need_weights=False)[0]
        x = x + self.dropout(attn_output)
        
        # Cross-attention sublayer with residual connection and gating
        x1 = self.layer_norm2(x)

        c = self.condition_norm(c)
        attn_output = self.cross_attn(x1, c, c, mask, need_weights=False)[0]
        x = x - F.relu(self.dropout(attn_output))

        # Feed-forward sublayer with residual connection
        x2 = self.layer_norm3(x)
        ff_output = self.feed_forward(x2)
        x = x + self.dropout(ff_output)
        
        return x

class IBExtractor(nn.Module):
    def __init__(self, embed_dim=256, in_channel=4, num_heads=8, ff_hidden_size=1024, num_layers=3, dropout=0, text_embed_dim=768, patient_info_size=3):
        super(IBExtractor, self).__init__()
        self.embed_dim = embed_dim
        self.signal_embedding = nn.Linear(in_channel, embed_dim) 
        self.RoPE = RoPEEmbedder(dim=embed_dim)
        # self.p_embedding = PatientInfoEmbedder(patient_info_size, embed_dim)
        # self.text_projector = nn.Linear(text_embed_dim, embed_dim)
        self.text_projector = nn.Linear(text_embed_dim + patient_info_size, embed_dim)

        self.layers = nn.ModuleList([
            IBEncoderLayer(embed_dim, num_heads, ff_hidden_size, dropout)
            for _ in range(num_layers)
        ])
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.classfree_emb = nn.Parameter(torch.ones([1, embed_dim]))


    # def extract_features(self, x, text_embed, mask, p, reduce=True):
    #     # Add positional encoding to input embeddings
    #     # (B, L, C) -> (B, L, embed_dim)
    #     x = self.signal_embedding(x)
    #     x = self.RoPE(x)

    #     # (B, L, text_embed_dim) -> (B, L, embed_dim)        
    #     text_embed = self.text_projector(text_embed)

    #     # (B, L) -> (B, L, embed_dim)
    #     p = self.p_embedding(p)
        
    #     # c: (B, L_e + L_p, dim)
    #     c = torch.concat([p, text_embed], dim=1)
    #     if mask is not None:
    #         mask = torch.concat([torch.ones(p.shape[0], p.shape[1], device=mask.device), mask], dim=1)

    #     # Pass input through each transformer encoder layer
    #     for layer in self.layers:
    #         x = layer(x, c, mask)
        
    #     if reduce:
    #         # (B, L, dim) -> (B, dim)
    #         return x.mean(dim=1)
    #     else:
    #         # (B, L, dim)
    #         return x

    def extract_features(self, x, text_embed, mask, p, reduce=True):
        # Add positional encoding to input embeddings
        # (B, L, C) -> (B, L, embed_dim)
        x = self.signal_embedding(x)
        x = self.RoPE(x)

        if text_embed is not None:
            p = p.unsqueeze(1).repeat(1, text_embed.shape[1], 1)# (B, L, D)
            # c: (B, L, text_embed_dim + pat_size)
            c = torch.concat([text_embed, p], dim=-1)

            # (B, L, text_embed_dim + pat_size) -> (B, L, embed_dim)        
            c = self.text_projector(c)

        else:
            # When condition is not applied
            c = self.classfree_emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
            mask = None

        # Pass input through each transformer encoder layer
        for layer in self.layers:
            x = layer(x, c, mask)
        
        if reduce:
            # (B, L, dim) -> (B, dim)
            return x.mean(dim=1)
        else:
            # (B, L, dim)
            return x

    def forward(self, x1, e1, m1, p1, x2, e2, m2, p2):
        features_1 = self.extract_features(x1, e1, m1, p1)
        features_2 = self.extract_features(x2, e2, m2, p2)

        # normalized features
        features_1 = features_1 / features_1.norm(dim=-1, keepdim=True)
        features_2 = features_2 / features_2.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_1 = logit_scale * features_1 @ features_2.t()
        logits_2 = logit_scale * features_2 @ features_1.t()

        # (B, B)
        return logits_1, logits_2


if __name__ == "__main__":
    model = IBExtractor(num_layers=3)
    x1 = torch.rand((16, 128, 4))
    x2 = torch.rand((16, 128, 4))
    e1 = torch.rand((16, 5, 768))
    e2 = torch.rand((16, 5, 768))
    p1 = torch.rand((16, 2))
    p2 = torch.rand((16, 2))
    m1 = torch.ones((16, 5))
    m2 = torch.ones((16, 5))

    # e1 = None
    # e2 = None
    output = model(x1, e1, m1, p1, x2, e2, m2, p2)
    print(output[0].shape)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters in the model: {total_params}')