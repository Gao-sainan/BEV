# ResNet + BiFPN
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from config import Config
c = Config()

class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    
    def __init__(self, block, layers) -> None:
        self.inplanes = 64
        super(ResNet, self).__init__()
        # 7x7 convolution, 64, stride=2
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # 3x3 max pool, stride=2
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # orignal resnet
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # multi-scale features used in BiFPN
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        
        # x = self.avgpool(c5)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        
        return [c2, c3, c4, c5]
    
class DepthwiseConvBlock(nn.Module):
    
    '''depthwise separable convolution, reduce the params
        Depthwise Conv + Pointwise Conv'''
        
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, 
                 padding=0, dilation=1, freeze_bn=False) -> None:
        super(DepthwiseConvBlock, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, 
                                   padding, dilation, groups=in_channels, bias=False)
        self.ponitwise = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                   padding, dilation, groups=1, bias=False)
        nn.init.kaiming_uniform_(self.depthwise.weight)
        nn.init.kaiming_uniform_(self.ponitwise.weight)
        
        self.bn = nn.BatchNorm2d(out_channels)
        nn.init.zeros_(self.bn.bias)
        nn.init.uniform_(self.bn.weight)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.ponitwise(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return x
    
class BiFPNBlock(nn.Module):
    
    '''Repeated Bi-directional Feature Pyramid Network Block'''

    def __init__(self, feature_size=88, epsilon=0.0001) -> None:
        super(BiFPNBlock, self).__init__()
        self.epsilon = epsilon
        
        self.p2_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p3_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p4_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p3_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p4_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p5_out = DepthwiseConvBlock(feature_size, feature_size)
        
        self.w1 = nn.Parameter(torch.ones((2, 3), requires_grad=True))
        self.w1_relu = nn.ReLU()
        self.w2 = nn.Parameter(torch.ones((3, 3), requires_grad=True))
        self.w2_relu = nn.ReLU()
        
    def forward(self, x):
        p2_x, p3_x, p4_x, p5_x = x
        
        # Fast normalized fusion
        # top-down
        w1 = self.w1_relu(self.w1)
        w1 = w1 / (torch.sum(w1, dim=0) + self.epsilon)
        w2 = self.w2_relu(self.w2)
        w2 = w2 / (torch.sum(w2, dim=0) + self.epsilon)
        
        p5_td = p5_x
        p4_td = self.p4_td(w1[0, 0] * p4_x + w1[1, 0] * F.interpolate(p5_td, scale_factor=2))
        p3_td = self.p3_td(w1[0, 1] * p3_x + w1[1, 1] * F.interpolate(p4_td, scale_factor=2))
        p2_td = self.p2_td(w1[0, 2] * p2_x + w1[1, 2] * F.interpolate(p3_td, scale_factor=2))
        
        # bottom-up
        p2_out = p2_td
        p3_out = self.p3_out(w2[0, 0] * p3_x + w2[1, 0] * p3_td + w2[2, 0] * nn.Upsample(scale_factor=0.5)(p2_out))
        p4_out = self.p4_out(w2[0, 1] * p4_x + w2[1, 1] * p4_td + w2[2, 1] * nn.Upsample(scale_factor=0.5)(p3_out))
        p5_out = self.p5_out(w2[0, 2] * p5_x + w2[1, 2] * p5_td + w2[2, 2] * nn.Upsample(scale_factor=0.5)(p4_out))
        
        return [p2_out, p3_out, p4_out, p5_out]
            
class BiFPN(nn.Module):
    '''extract multiple scale features'''
    def __init__(self, size, feature_size=88, num_layers=2, epsilon=0.0001) -> None:
        super(BiFPN, self).__init__()
        self.p2 = nn.Conv2d(size[0], feature_size, kernel_size=1, stride=1, padding=0)
        self.p3 = nn.Conv2d(size[1], feature_size, kernel_size=1, stride=1, padding=0)
        self.p4 = nn.Conv2d(size[2], feature_size, kernel_size=1, stride=1, padding=0)
        self.p5 = nn.Conv2d(size[3], feature_size, kernel_size=1, stride=1, padding=0)
        nn.init.xavier_normal_(self.p2.weight)
        nn.init.xavier_normal_(self.p3.weight)
        nn.init.xavier_normal_(self.p4.weight)
        nn.init.xavier_normal_(self.p5.weight)
        
        
        bifpn = []
        for _ in range(num_layers):
            bifpn.append(BiFPNBlock(feature_size))
        self.bifpn = nn.Sequential(*bifpn)
        
    def forward(self, x):
        c2, c3, c4, c5 = x
        p2_x = self.p2(c2)
        p3_x = self.p3(c3)
        p4_x = self.p4(c4)
        p5_x = self.p5(c5)
        
        features = [p2_x, p3_x, p4_x, p5_x]
        return self.bifpn(features)

class MultiScaleFeature(nn.Module):
    def __init__(self) -> None:
        super(MultiScaleFeature, self).__init__()

        self.resnet = ResNet(BottleNeck, [3, 4, 6, 3]).to(c.device)
        # initialize resnet with ImageNet Pretrained weight
        resnet50 = models.resnet50(pretrained=True)
        model_dict = self.resnet.state_dict()
        pretrained_dict = resnet50.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.resnet.load_state_dict(model_dict)
        
        self.bifpn = BiFPN([256, 512, 1024, 2048]).to(c.device)

    def forward(self, x):
        x = self.resnet(x)
        x = self.bifpn(x)
        return x

