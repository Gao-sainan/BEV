
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



# visualize feature maps

config = Config()

transform_train = transforms.Compose([ 
                                transforms.ToTensor(),
                                # transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
                                ])

      
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


train = ReplicaDataset(config.train_dir, transform_train)

train_loader = DataLoader(train, batch_size=2, shuffle=True)



img1, img2, labels= next(iter(train_loader))
_, labels = torch.max(labels, 1)
plt.figure()
for i in range(1,32):
    plt.subplot(4,8,i)
    plt.imshow([i-1])
    plt.xticks([])
    plt.yticks([])
plt.show()
    
plt.savefig("tw_images.jpg")
    


