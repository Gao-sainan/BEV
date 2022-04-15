
import cv2 as cv
from torch.utils.data import Dataset
import os
import torch
import matplotlib.pyplot as plt



class ReplicaDataset(Dataset):
    def __init__(self, root, transform=None):
        self.data_info = self.get_data_info(root)
        self.transform = transform
    
    def __len__(self):
        return len(self.data_info)
        
    def __getitem__(self, index):
        path_image1, path_image2, path_label = self.data_info[index]
        image1 = cv.imread(path_image1)
        image1 = cv.cvtColor(image1, cv.COLOR_BGR2RGB)
        image2 = cv.imread(path_image2)
        image2 = cv.cvtColor(image2, cv.COLOR_BGR2RGB)
        label = cv.imread(path_label, cv.IMREAD_GRAYSCALE)
        
        
        
        raster = torch.zeros(1, 60, 80)
        
        label = torch.from_numpy(label)
        label = self.label_one_hot(label)
        
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2, raster, label
    
    def label_one_hot(self, label):
        one_hot = []
        for i in [0, 255]:
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
   



        
    
    
        