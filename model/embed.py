# patch embedding & positional encoding
import math

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):
    '''linear projection
        (32 * 3)-d vector (same with PE)'''
    def __init__(self, patch_height, patch_width, patch_dim, embed_dim=32*2) -> None:
        super(PatchEmbedding, self).__init__()
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, embed_dim)
            )
        
    def forward(self, x):
        x = self.to_patch_embedding(x)
        return x
    
class FeaturesEmbedding(nn.Module):
    def __init__(self, cam_id, patch_height_l=10, patch_width_l=10, patch_height_s=5, patch_width_s=5, feature_size=88, embed_dim=32) -> None:
        super(FeaturesEmbedding, self).__init__()
        self.patch_height_l = patch_height_l
        self.patch_width_l = patch_width_l
        self.patch_height_s = patch_height_s
        self.patch_width_s = patch_width_s
        self.cam_id = cam_id
        self.embed_dim = embed_dim
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        
        
        self.patch_embedding_l = PatchEmbedding(patch_height_l, patch_width_l, patch_dim=patch_height_l*patch_width_l*feature_size)
        self.patch_embedding_s = PatchEmbedding(patch_height_s, patch_width_s, patch_dim=patch_height_s*patch_width_s*feature_size)
        # self.pos_encoding = PositionalEncoding(cam_id)
        
    def positional_encoding(self, patch_num):
        # 2 input images and a output raster
        cam_encoding = np.zeros((3, self.embed_dim))
        for cam in range(cam_encoding.shape[0]):
            for j in range(cam_encoding.shape[1]):
                cam_encoding[cam][j] = math.sin(cam / (10000 ** (2*j/self.embed_dim))) if j % 2 == 0 else math.cos(cam / (10000 ** (2*j/self.embed_dim)))
        
        positinal_encoding = np.zeros((patch_num, self.embed_dim * 2))
        for pos in range(positinal_encoding.shape[0]):
            for i in range(self.embed_dim):
                positinal_encoding[pos][i] = math.sin(pos / (10000 ** (2*i/self.embed_dim))) if i % 2 == 0 else math.cos(pos / (10000 ** (2*i/self.embed_dim)))
            positinal_encoding[pos][self.embed_dim:] = cam_encoding[self.cam_id]
        positinal_encoding = torch.from_numpy(positinal_encoding).to(self.device)
        
        return positinal_encoding
        
        
    def forward(self, x):
        p2_x, p3_x, p4_x, p5_x = x
        
        p2_patch_num = (p2_x.shape[2] // self.patch_height_l) * (p2_x.shape[3] // self.patch_width_l)
        p3_patch_num = (p3_x.shape[2] // self.patch_height_l) * (p3_x.shape[3] // self.patch_width_l)
        p4_patch_num = (p4_x.shape[2] // self.patch_height_s) * (p4_x.shape[3] // self.patch_width_s)
        p5_patch_num = (p5_x.shape[2] // self.patch_height_s) * (p5_x.shape[3] // self.patch_width_s)
        
        p2_patch_embed = self.patch_embedding_l(p2_x)
        p3_patch_embed = self.patch_embedding_l(p3_x)
        p4_patch_embed = self.patch_embedding_s(p4_x)
        p5_patch_embed = self.patch_embedding_s(p5_x)
        
        p2_pos_encode = self.positional_encoding(p2_patch_num)
        p3_pos_encode = self.positional_encoding(p3_patch_num)
        p4_pos_encode = self.positional_encoding(p4_patch_num)
        p5_pos_encode = self.positional_encoding(p5_patch_num)
        
        p2_embed = p2_patch_embed + p2_pos_encode
        
        p3_embed = p3_patch_embed + p3_pos_encode
        
        p4_embed = p4_patch_embed + p4_pos_encode
        
        p5_embed = p5_patch_embed + p5_pos_encode
        
        
        return [p2_embed, p3_embed, p4_embed, p5_embed]
        
        
    
