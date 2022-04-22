from turtle import forward
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from config import Config
from einops.layers.torch import Rearrange, repeat

c = Config()

class PatchEmbedding(nn.Module):
    '''linear projection
        (32 * 3)-d vector (same with PE)'''
    def __init__(self, embed_dim, patch_height=c.patch_height, patch_width=c.patch_width, patch_dim=88) -> None:
        super(PatchEmbedding, self).__init__()
        self.to_patch_embedding = nn.Sequential(
            # (b, c, patch_num, patch_dim)
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            # (b, c, patch_num, patch_dim) -> (b, c, patch_num, embed_dim)
            nn.Linear(patch_dim, embed_dim)
            )
        
        for m in self.to_patch_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.)

        
    def forward(self, x):
        x = self.to_patch_embedding(x)
        return x
    
class PostionalEncoding(nn.Module):
    def __init__(self, pe_dim, x, y):
        super(PostionalEncoding, self).__init__()
        self.encoding_x = torch.zeros(x, pe_dim).to(c.device)
        self.encoding_x.requires_grad = False
        self.encoding_y = torch.zeros(y, pe_dim).to(c.device)
        self.encoding_y.requires_grad = False
        pos_x = torch.arange(0, x).to(c.device)
        pos_x = pos_x.float().unsqueeze(dim=1)
        pos_y = torch.arange(0, y).to(c.device)
        pos_y = pos_y.float().unsqueeze(dim=1)
        _2i = torch.arange(0, pe_dim // 2).float().to(c.device)

        self.encoding_x[:, :pe_dim // 2] = torch.sin(pos_x / (10000 ** (_2i / pe_dim)))
        self.encoding_x[:, pe_dim // 2:] = torch.cos(pos_x / (10000 ** (_2i / pe_dim)))
        self.encoding_y[:, :pe_dim // 2] = torch.sin(pos_y / (10000 ** (_2i / pe_dim)))
        self.encoding_y[:, pe_dim // 2:] = torch.cos(pos_y / (10000 ** (_2i / pe_dim)))
        
        self.encoding = torch.zeros(x*y, pe_dim * 2).to(c.device)
        self.encoding.requires_grad = False
        for i in range(self.encoding.shape[0]):
            idx = i % x
            idy = i // x
            self.encoding[i,:] = torch.cat((self.encoding_x[idx], self.encoding_y[idy]), dim=-1)

    def forward(self, x):
        batch_size, features, h, w = x.size()
        patch_num = h * w
        # (b, patch_num, patch_dim) -> (b, patch_num, pe_dim(=embed_dim))
        return self.encoding[:patch_num, :]

class Embedding(nn.Module):
    def __init__(self, h, w) -> None:
        super(Embedding, self).__init__()

        self.patch_embed = PatchEmbedding(96)
        self.cam_pos_encode = nn.Parameter(torch.randn(2, 32))
        self.cam_pos_encode.requires_grad = True
        self.input_pos_encode = PostionalEncoding(32, h, w)
        
    def forward(self, x):
        f1, f2 = x
        patch_embed1 = self.patch_embed(f1)
        patch_embed2 = self.patch_embed(f2)
        pos_encode1 = self.input_pos_encode(f1)
        pos_encode2 = self.input_pos_encode(f2)
        cam_pos = self.cam_pos_encode

        # (b, patch_num(300), 64)
        cam0 = repeat(cam_pos[0], 'd -> (repeat r) d', r =1, repeat=x.size()[-1] * x.size()[-2])
        cam1 = repeat(cam_pos[1], 'd -> (repeat r) d', r =1, repeat=x.size()[-1] * x.size()[-2])

        embed1 = patch_embed1 + torch.cat((pos_encode1, cam0), dim=-1)
        embed2 = patch_embed2 + torch.cat((pos_encode2, cam1), dim=-1)
        
        return embed1, embed2


class MultiHead_Attention(nn.Module):
    '''Multi-head Attention (cross attention)'''
    def __init__(self, ma_dim=c.ma_dim, head_dim=c.head_dim, n_heads=c.n_heads, dropout=c.dropout, embed_dim=c.embed_dim):
        super(MultiHead_Attention, self).__init__()
        self.ma_dim = ma_dim
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        
        self.to_q = nn.Linear(embed_dim, ma_dim)
        self.to_k = nn.Linear(96, ma_dim)
        self.to_v = nn.Linear(96, ma_dim)
        
        
        self.softmax = nn.Softmax(dim=-1)
        
        self.to_out = nn.Sequential(
            nn.Linear(ma_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.scale = 1 / np.sqrt(head_dim)
        
    def forward(self, raster, image):
        # raster, image = x
        q = self.to_q(raster)
        k1 = self.to_k(image)
        v1 = self.to_v(image)
        
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.n_heads)
        k1 = rearrange(k1, 'b n (h d) -> b h n d', h = self.n_heads)
        v1 = rearrange(v1, 'b n (h d) -> b h n d', h = self.n_heads)
        
        dots1 = torch.matmul(q, k1.transpose(-1, -2)) * self.scale
        atten1 = self.softmax(dots1)
        
        out1 = torch.matmul(atten1, v1)
        out1 = rearrange(out1, 'b h n d -> b n (h d)')
        
        
        out = self.to_out(out1)
        
        return out

class FeedForward(nn.Module):
    '''tow layers with acitivate function of GELU or RELU'''
    def __init__(self, embed_dim=c.embed_dim, hidden_dim=c.hidden_dim, dropout=c.dropout):
        super(FeedForward, self).__init__()
    
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        ).to(c.device)
        
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.zeros_(m.bias)
                
        
    def forward(self, x):
        return self.mlp(x)

    
class Add_Norm(nn.Module):
    '''residual and layer norm'''
    def __init__(self, embed_dim=32, dropout=0.1):
        super(Add_Norm, self).__init__()
        self.norm  = nn.LayerNorm(embed_dim).to(c.device)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sub_layer, **kwargs):
        sub_out = sub_layer(x, **kwargs)
        x = self.dropout(x + sub_out)
        return self.norm(x)

    
class TransfomerLayer(nn.Module):
    '''repeated block of MHA & MLP & ADD_NORM stack together '''
    def __init__(self, embed_dim, ma_dim, head_dim, n_heads, hidden_dim, dropout):
        super(TransfomerLayer, self).__init__()

        self.multi_atten = MultiHead_Attention(ma_dim, head_dim, n_heads, dropout, embed_dim).to(c.device)
        self.feed_forward = FeedForward(embed_dim, hidden_dim, dropout).to(c.device)
        self.add_norm = Add_Norm(embed_dim, dropout).to(c.device)
        
        for m in self.multi_atten.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.zeros_(m.bias)
        
    def forward(self, x):
        raster, image1, image2 = x
        x1 = self.add_norm(raster, self.multi_atten, image=image1)
        x2 = self.add_norm(raster, self.multi_atten, image=image2)
        x = self.add_norm(x1+x2, self.feed_forward)
        return (x, image1, image2)
    
class Transformer(nn.Module):

    def __init__(self, N, embed_dim=c.embed_dim, ma_dim=c.ma_dim, head_dim=c.head_dim, n_heads=c.n_heads, hidden_dim=c.hidden_dim, dropout=c.dropout):
        super(Transformer, self).__init__()
       
        self.encoder = nn.Sequential(*[TransfomerLayer(embed_dim, ma_dim, head_dim, n_heads, hidden_dim, dropout) for _ in range(N)]).to(c.device)
        
    def forward(self, x):
        output, _ , _= self.encoder(x)
        return output
    