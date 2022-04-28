import torch
import torch.nn as nn
from einops import rearrange, repeat
from config import Config

from features_extractor import BiFPN, BottleNeck, ResNet, MultiScaleFeature
from cross_attention import PostionalEncoding, PatchEmbedding, Transformer

c = Config()
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels , kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        
        for m in self.double_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.zeros_(m.bias)
                

    def forward(self, x):
        return self.double_conv(x)
    
      
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
            

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        
        return x

class SegHead(nn.Module):
    def __init__(self, out_channels, in_channels=64,) -> None:
        super(SegHead, self).__init__()
        
        self.up1 = Up(in_channels, in_channels // 2)
        self.up2 = Up(in_channels // 2, in_channels // 4)
        
        self.out = nn.Conv2d(in_channels // 4, out_channels, kernel_size=1)

        nn.init.kaiming_uniform_(self.out.weight)
        nn.init.constant_(self.out.bias, 0.)
        
    def forward(self, x):
        
        x = x.view(x.size()[0], x.size()[-1], 60, 80)
        
        x = self.up1(x)
        x = self.up2(x)
        
        x = self.out(x)
        
        return x
        
class Bev(nn.Module):
    def __init__(self, n_transformer) -> None:
        super(Bev, self).__init__()
        
        self.multiscalefeature1 = MultiScaleFeature()
        self.multiscalefeature2 = MultiScaleFeature()


        # input(b, 88, 15, 20) -> (b, 15*20, 88) -> (b, 300, 96)
        self.patch_embed = PatchEmbedding(64)
        self.cam_pos_encode = nn.Parameter(torch.randn(2, 64))
        self.cam_pos_encode.requires_grad = True
        self.input_pos_encode = PostionalEncoding(32, 15, 20)
        
        # raster(b, 1, 60, 80) -> (b, 60*80, 1) -> (b, 4800, 64)
        self.raster_patch_embed = PatchEmbedding(64, patch_dim=1)
        self.raster_pos_encode = PostionalEncoding(32, 60, 80)

        # pool mutil-scale features to (C, 1, 1)
        self.raster_pool1 = nn.MaxPool2d(kernel_size=(118, 158))
        self.raster_pool2 = nn.MaxPool2d(kernel_size=(58, 78))
        self.raster_pool3 = nn.MaxPool2d(kernel_size=(28, 38))
        self.raster_pool4 = nn.MaxPool2d(kernel_size=(13, 18))
        
        # conv mutil-scale features to raster size (4800, 1, 1)
        self.raster_conv = nn.Conv2d(88*4, 4800, kernel_size=1)
        nn.init.kaiming_uniform_(self.raster_conv.weight)
        
        
        self.T = Transformer(n_transformer).to(c.device)
        
        self.seg = SegHead(2).to(c.device)

        
    def forward(self, input1, input2):
        # input1, input2 = x

        f1_p2, f1_p3, f1_p4, f1_p5 = self.multiscalefeature1(input1)
        f2_p2, f2_p3, f2_p4, f2_p5 = self.multiscalefeature2(input2)

        # (b, 88, 1, 1)
        c2  = self.raster_pool1(f1_p2)
        c3  = self.raster_pool2(f1_p3)
        c4  = self.raster_pool3(f1_p4)
        c5  = self.raster_pool4(f1_p5)
        context1 = torch.cat((c2, c3, c4, c5), dim=1)

        c2_  = self.raster_pool1(f2_p2)
        c3_  = self.raster_pool2(f2_p3)
        c4_  = self.raster_pool3(f2_p4)
        c5_  = self.raster_pool4(f2_p5)
        context2 = torch.cat((c2_, c3_, c4_, c5_), dim=1)

        # (b, 4800, 1, 1)
        context1 = self.raster_conv(context1)
        context2 = self.raster_conv(context2)
        # (b, 1, 60, 80) same as raster size
        context1 = rearrange(context1, 'b (h w) p1 p2 -> b (p1 p2) h w', h=60, w =80)
        context2 = rearrange(context2, 'b (h w) p1 p2 -> b (p1 p2) h w', h=60, w =80)
        context_summary = context1 + context2

        multi_features1_embed = self.patch_embed(f1_p5)
        multi_features2_embed = self.patch_embed(f2_p5)
        pos_encode1 = self.input_pos_encode(f1_p5)
        pos_encode2 = self.input_pos_encode(f2_p5)
        cam_pos = self.cam_pos_encode

        # (b, patch_num(300), 64)
        cam0 = repeat(cam_pos[1], 'd -> (repeat r) d', r =1, repeat=300)
        cam1 = repeat(cam_pos[0], 'd -> (repeat r) d', r =1, repeat=300)

        # embed1 = multi_features1_embed + torch.cat((pos_encode1, cam0), dim=-1)
        # embed2 = multi_features2_embed + torch.cat((pos_encode2, cam1), dim=-1)
        
        embed1 = multi_features1_embed + pos_encode1 + cam0
        embed2 = multi_features2_embed + pos_encode2 + cam1

        # raster(b, 1, 60, 80) -> (b, 60*80, 1) -> (b, 4800, 32)
        raster_embed = self.raster_patch_embed(context_summary)
        raster_pos = self.raster_pos_encode(context_summary)
        embed_ = raster_embed + raster_pos
        
        # fusion_features = torch.cat((embed1, embed2), dim=1)
        # transformer_input = (embed_.to(torch.float32), fusion_features.to(torch.float32))
        # out= self.T(transformer_input)
        
        predict = self.seg(self.T((embed_, torch.cat((embed1, embed2), dim=1))))
        return predict
    

    


