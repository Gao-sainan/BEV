
import cv2 as cv
from bev import Bev
from config import Config
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from data_function import ReplicaDataset
from torch.utils.data import DataLoader
from torchvision import models
from features_extractor import BiFPN, BottleNeck, ResNet, MultiScaleFeature
import matplotlib.pyplot as plt
from einops import rearrange
# from torchstat import stat

c = Config()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([ 
                                transforms.ToTensor(),
                                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
                                ])

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


test = ReplicaDataset(c.test_dir, transform)
test_loader = DataLoader(
    test, 
    batch_size=c.BATCH_SIZE, 
    num_workers=8)

model_path = c.cp_dir

model = Bev(c.N)
model.load_state_dict(torch.load(model_path))
model.to(device)

children = list(model.children())
multiscalefeature1 = children[0]
multiscalefeature2 = children[1]
multiscalefeature1.eval()
multiscalefeature2.eval()
for i, data in enumerate(test_loader):

    img1, img2, labels = data
    
    img1 = img1.to(device)
    img2 = img2.to(device)
    labels = labels.to(device)

    msf1 = multiscalefeature1(img1)
    msf2 = multiscalefeature2(img2)
    vis1 = rearrange(msf1[0][0, :30], '(c1 c2) h w -> (c1 h) (c2 w)', c1=5, c2=6)
    vis2 = rearrange(msf2[0][0, :30], '(c1 c2) h w -> (c1 h) (c2 w)', c1=5, c2=6)

    cv.imwrite('multifeature_vis1_o.png', vis1.detach().cpu().numpy()*255)
    cv.imwrite('multifeature_vis2_o.png', vis2.detach().cpu().numpy()*255)



