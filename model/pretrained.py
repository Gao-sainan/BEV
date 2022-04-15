
from features_extractor import BiFPN, BottleNeck, ResNet
from torch.utils.tensorboard import SummaryWriter 
import torch
import torch.nn as nn


class resbifpn(nn.Module):
    def __init__(self) -> None:
        super(resbifpn, self).__init__()
        self.resnet = ResNet(BottleNeck, [3, 4, 6, 3])
        self.bifpn = BiFPN([256, 512, 1024, 2048])
        
    def forward(self, x):
        x = self.resnet(x)
        x = self.bifpn(x)
        
        return x
    
input = torch.randn(1, 3, 480, 640)
model = resbifpn()
out = model(input)
for o in out:
    print(out.shape)

