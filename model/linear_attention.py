import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from config import Config
from einops.layers.torch import Rearrange
import math

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
    
def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

class LinformerSelfAttention(nn.Module):
    def __init__(self, seq_len=600, dim=c.embed_dim, k=32, heads=c.n_heads, dim_head=c.head_dim, one_kv_head = False, share_kv = False, dropout = 0.1):
        super(LinformerSelfAttention, self).__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'

        self.seq_len = seq_len
        self.k = k

        self.heads = heads

        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, dim_head * heads, bias = False)

        kv_dim = dim_head if one_kv_head else (dim_head * heads)
        self.to_k = nn.Linear(dim, kv_dim, bias = False)
        self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.share_kv = share_kv
        if not share_kv:
            self.to_v = nn.Linear(dim, kv_dim, bias = False)
            self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim_head * heads, dim)

    def forward(self, x, context, **kwargs):
        b, n, d, d_h, h, k = *x.shape, self.dim_head, self.heads, self.k

        kv_len = n if context is None else context.shape[1]
        assert kv_len == self.seq_len, f'the sequence length of the key / values must be {self.seq_len} - {kv_len} given'

        queries = self.to_q(x)

        proj_seq_len = lambda args: torch.einsum('bnd,nk->bkd', *args)

        kv_input = x if context is None else context

        keys = self.to_k(kv_input)
        values = self.to_v(kv_input) if not self.share_kv else keys

        kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)

        # project keys and values along the sequence length dimension to k

        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

        # merge head into batch for queries and key / values

        queries = queries.reshape(b, n, h, -1).transpose(1, 2)

        merge_key_values = lambda t: t.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        keys, values = map(merge_key_values, (keys, values))

        # attention

        dots = torch.einsum('bhnd,bhkd->bhnk', queries, keys) * (d_h ** -0.5)
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhnk,bhkd->bhnd', attn, values)

        # split heads
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


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

        self.lin_atten = LinformerSelfAttention().to(c.device)
        self.feed_forward = FeedForward(embed_dim, hidden_dim, dropout).to(c.device)
        self.add_norm = Add_Norm(embed_dim, dropout).to(c.device)
        
    def forward(self, x):
        raster, image = x
        x = self.add_norm(raster, self.lin_atten, context=image)
        x = self.add_norm(x, self.feed_forward)
        # self-atten or cross atten?
        return (x, image)
    
class Transformer(nn.Module):

    def __init__(self, N, embed_dim=c.embed_dim, ma_dim=c.ma_dim, head_dim=c.head_dim, n_heads=c.n_heads, hidden_dim=c.hidden_dim, dropout=c.dropout):
        super(Transformer, self).__init__()
       
        self.encoder = nn.Sequential(*[TransfomerLayer(embed_dim, ma_dim, head_dim, n_heads, hidden_dim, dropout) for _ in range(N)]).to(c.device)
        
    def forward(self, x):
        output, _ = self.encoder(x)
        return output
    