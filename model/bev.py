import torch
import torch.nn as nn

from einops import rearrange
from torchvision import models

from features_extractor import BiFPN, BottleNeck, ResNet
from cross_attention import PositionalEncoding, FeaturesEmbedding, Transformer


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        
        return self.conv(x)

class SegHead(nn.Module):
    def __init__(self, in_channels, out_channels, N=2) -> None:
        super(SegHead, self).__init__()
        
        self.up = nn.Sequential(*[Up(in_channels, in_channels) for _ in range(N)])
        self.out = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        
    def forward(self, x):
        x = rearrange(x, 'b (h w) (c) -> b c h w', h=60, w=80)
        x = self.up(x)
        
        return self.out(x)
        
class Bev(nn.Module):
    def __init__(self, cls, layers=[3, 4, 6, 3]) -> None:
        super(Bev, self).__init__()
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        
        self.resnet = ResNet(BottleNeck, layers).to(self.device)
        
        # initialize resnet with ImageNet Pretrained weight
        resnet50 = models.resnet50(pretrained=True)
        model_dict = self.resnet.state_dict()
        pretrained_dict = resnet50.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.resnet.load_state_dict(model_dict)
        
        self.bifpn = BiFPN([256, 512, 1024, 2048]).to(self.device)
        self.embed1 = FeaturesEmbedding(1).to(self.device)
        self.embed2 = FeaturesEmbedding(2).to(self.device)
        self.raster_pos_encode = FeaturesEmbedding(0).to(self.device)
        # pool mutil-scale features to (C, 1, 1)
        self.raster_pool1 = nn.AvgPool2d(kernel_size=(118, 158), stride=2, padding=0)
        self.raster_pool2 = nn.AvgPool2d(kernel_size=(58, 78), stride=2, padding=0)
        self.raster_pool3 = nn.AvgPool2d(kernel_size=(28, 38), stride=2, padding=0)
        self.raster_pool4 = nn.AvgPool2d(kernel_size=(13, 28), stride=2, padding=0)
        
        # TODO:conv mutil-scale features to raster size (4800, 1, 1)
        self.raster_conv = nn.Conv2d(, 4800, kernel_size=)
        
        
        self.T = Transformer().to(self.device)
        self.seg = SegHead(64, cls).to(self.device)
        
    def forward(self, input1, input2, output_raster):
        # input1, input2, output_raster = x
        # use the same resnet and bifpn for two input images

        res_out1 = self.resnet(input1)
        f1_p2, f1_p3, f1_p4, f1_p5 = self.bifpn(res_out1)
        res_out2 = self.resnet(input2)
        f2_p2, f2_p3, f2_p4, f2_p5 = self.bifpn(res_out2)
        
        multi_features1_embed = self.embed1(f1_p5)
        multi_features2_embed = self.embed2(f2_p5)
        
        raster_pos_encode = self.raster_pos_encode.positional_encoding(output_raster.shape[-1] * output_raster.shape[-2])
        raster_pos_encode = raster_pos_encode.reshape(1, raster_pos_encode.shape[0], raster_pos_encode.shape[1])
        
        fusion_features = torch.cat((multi_features1_embed, multi_features2_embed), dim=1)
        transformer_input = (raster_pos_encode.to(torch.float32), fusion_features.to(torch.float32))
        out= self.T(transformer_input)
        
        # fusion_features_list = []
        # out_list = []
        # for i in range(len(multi_features1_embed)):
        #     fusion_features = torch.cat((multi_features1_embed[i], multi_features2_embed[i]), dim=1)
        #     fusion_features_list.append(fusion_features)
            
        #     transformer_input = (raster_pos_encode.to(torch.float32), fusion_features.to(torch.float32))
        #     T = Transformer()
        #     out, _ = T(transformer_input)
        #     out_list.append(out)
        # out_cat = torch.cat((out_list), dim=-1)
        
        predict = self.seg(out)
        return predict
    
    
    


