import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

class PatchEmbedding(nn.Module):
    '''linear projection
        (32 * 3)-d vector (same with PE)'''
    def __init__(self, embed_dim, patch_height=1, patch_width=1, patch_dim=88) -> None:
        super(PatchEmbedding, self).__init__()
        self.to_patch_embedding = nn.Sequential(
            # (b, c, patch_num, patch_dim)
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            # (b, c, patch_num, patch_dim) -> (b, c, patch_num, embed_dim)
            nn.Linear(patch_dim, embed_dim)
            )
        
    def forward(self, x):
        x = self.to_patch_embedding(x)
        return x
    
class PostionalEncoding(nn.Module):
    def __init__(self, pe_dim, patch_num, device):
        super(PostionalEncoding, self).__init__()
        self.encoding = torch.zeros(patch_num, pe_dim, device=device)
        self.encoding.requires_grad = False
        pos = torch.arange(0, patch_num, device=device)
        pos = pos.float().unsqueeze(dim=1)
        _2i = torch.arange(0, pe_dim, step=2, device=device).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / pe_dim)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / pe_dim)))

    def forward(self, x):
        batch_size, features, h, w = x.size()
        patch_num = h * w
        # (b, patch_num, patch_dim) -> (b, patch_num, pe_dim(=embed_dim))
        return self.encoding[:patch_num, :]

    
class FeaturesEmbedding(nn.Module):
    def __init__(self, cam_id, patch_height=1, patch_width=1, feature_size=88, embed_dim=32) -> None:
        super(FeaturesEmbedding, self).__init__()
        self.patch_height = patch_height
        self.patch_width = patch_width

        self.cam_id = cam_id
        self.embed_dim = embed_dim
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        
        self.patch_embedding = PatchEmbedding(self.patch_height, self.patch_width, patch_dim=patch_height*patch_width*feature_size).to(self.device)
        self.pos_encoding = PostionalEncoding(32, 5).to(self.device)
        self.cam_pos = PostionalEncoding(32, 2).to(self.device)

    
    def forward(self, x):
 
        patch_num = (x.size()[2] // self.patch_height) * (x.size()[3] // self.patch_width)
        patch_embed = self.patch_embedding(x)
        pos_encode = self.PositionalEncoding(patch_num).to(self.device)
        embed_out = patch_embed + pos_encode
        
        return embed_out

class MultiHead_Attention(nn.Module):
    '''Multi-head Attention (cross attention)'''
    def __init__(self, ma_dim=256, head_dim=32, n_heads=8, dropout=0.1, embed_dim=32):
        super(MultiHead_Attention, self).__init__()
        self.ma_dim = ma_dim
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        
        self.to_q = nn.Linear(embed_dim, ma_dim)
        self.to_k = nn.Linear(embed_dim, ma_dim)
        self.to_v = nn.Linear(embed_dim, ma_dim)
        
        self.a = nn.Softmax(dim=-1)
        
        self.to_out = nn.Sequential(
            nn.Linear(ma_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.scale = 1 / np.sqrt(head_dim)
        
    def forward(self, raster, image):
        # raster, image = x
        q = self.to_q(raster)
        k = self.to_k(image)
        v = self.to_v(image)
        
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.n_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h = self.n_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h = self.n_heads)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        atten = self.a(dots)
        
        out = torch.matmul(atten, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)

class FeedForward(nn.Module):
    '''tow layers with acitivate function of GELU or RELU'''
    def __init__(self, embed_dim=32, hidden_dim=512, dropout=0.1):
        super(FeedForward, self).__init__()
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        ).to(self.device)
        
    def forward(self, x):
        return self.mlp(x)

    
class Add_Norm(nn.Module):
    '''residual and layer norm'''
    def __init__(self, embed_dim=32, dropout=0.1):
        super(Add_Norm, self).__init__()
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.norm  = nn.LayerNorm(embed_dim).to(self.device)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sub_layer, **kwargs):
        sub_out = sub_layer(x, **kwargs)
        x = self.dropout(x + sub_out)
        return self.norm(x)

    
class TransfomerLayer(nn.Module):
    '''repeated block of MHA & MLP & ADD_NORM stack together '''
    def __init__(self, embed_dim, ma_dim, head_dim, n_heads, hidden_dim, dropout=0.1):
        super(TransfomerLayer, self).__init__()
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.multi_atten = MultiHead_Attention(ma_dim, head_dim, n_heads, dropout, embed_dim).to(self.device)
        self.feed_forward = FeedForward(embed_dim, hidden_dim, dropout).to(self.device)
        self.add_norm = Add_Norm(embed_dim, dropout).to(self.device)
        
    def forward(self, x):
        raster, image = x
        x = self.add_norm(raster, self.multi_atten, image=image)
        x = self.add_norm(x, self.feed_forward)
        return (x, image)
    
class Transformer(nn.Module):

    def __init__(self, embed_dim=32, ma_dim=256, head_dim=32, n_heads=8, hidden_dim=512, dropout=0.1, N=6):
        super(Transformer, self).__init__()
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        
        self.encoder = nn.Sequential(*[TransfomerLayer(embed_dim, ma_dim, head_dim, n_heads, hidden_dim, dropout) for _ in range(N)]).to(self.device)
        
    def forward(self, x):
        output, _ = self.encoder(x)
        return output
    