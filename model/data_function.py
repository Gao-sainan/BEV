
from PIL import Image
from torch.utils.data import Dataset
import os
import torch
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
        listdir = os.listdir(root)
        image_path_list = list(filter(lambda x: x.startswith('images'), listdir))
        image_path_list.sort()
        label_path_list = list(filter(lambda x: x.startswith('label'), listdir))
        label_path_list.sort()
        full_data = []
        for p in range(len(image_path_list)):
            image_path = image_path_list[p]
            label_path = label_path_list[p]
            image_names = os.listdir(os.path.join(root, image_path))
            label_names = os.listdir(os.path.join(root, label_path))
            image_names = list(filter(lambda x: x.endswith('.jpg'), image_names))
            label_names = list(filter(lambda x: x.endswith('.png'), label_names))
            image_names.sort()
            label_names.sort()
        
            data_info = []
            for i in range(0, len(image_names), 2):
                image1 = image_names[i]
                image2 = image_names[i+1]
                label = label_names[i//2]

                path_image1 = os.path.join(root, image_path, image1)
                path_image2 = os.path.join(root, image_path, image2)
                path_label = os.path.join(root, label_path, label)

                data_info.append((path_image1, path_image2, path_label))
            full_data.extend(data_info)
        return full_data




        
    
    
        