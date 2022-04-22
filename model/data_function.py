from PIL import Image
import cv2 as cv
from cv2 import transform
from torch.utils.data import Dataset
import os
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms




class ReplicaDataset(Dataset):
    def __init__(self, root, transform=None):
        self.data_info = self.get_data_info(root)
        self.transform = transform
    
    def __len__(self):
        return len(self.data_info)
        
    def __getitem__(self, index):
        path_image1, path_image2, path_label = self.data_info[index]
        image1 = Image.open(path_image1).convert('RGB')
        image2 = Image.open(path_image2).convert('RGB')
        label = Image.open(path_label)
        
        label = transforms.functional.to_tensor(label)
        label = self.label_one_hot(label)
        raster = torch.zeros(size=(1, 60, 80))
        
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)


        return image1, image2, label
    
    def label_one_hot(self, label):
        label = label.squeeze(0)
        one_hot = []
        for i in range(2):
            temp_prob = label==i 
            one_hot.append(temp_prob.unsqueeze(0))
        output = torch.cat(one_hot, dim=0)
        return output.to(torch.float32) 
         
    @staticmethod
    def get_data_info(root):
        image_path = os.path.join(root, 'images/')
        label_path = os.path.join(root, 'label/')
        image_names = os.listdir(image_path)
        label_names = os.listdir(label_path)
        image_names = list(filter(lambda x: x.endswith('.jpg'), image_names))
        label_names = list(filter(lambda x: x.endswith('.png'), label_names))
        image_names.sort()
        label_names.sort()
        
        data_info = []
        for i in range(0, len(image_names), 2):
            image1 = image_names[i]
            image2 = image_names[i+1]
            label = label_names[i//2]

            path_image1 = os.path.join(root, 'images/', image1)
            path_image2 = os.path.join(root, 'images/', image2)
            path_label = os.path.join(root, 'label/', label)

            data_info.append((path_image1, path_image2, path_label))
        return data_info
   



        
    
    
        