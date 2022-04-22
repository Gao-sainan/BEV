from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter   
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from config import Config
from data_function import ReplicaDataset
from bev import Bev
import torchvision
import cv2 as cv
import os
from metrics import iou_pytorch
from torch.utils.data import random_split
from torchmetrics import ConfusionMatrix, JaccardIndex



config = Config()
    
transform = transforms.Compose([ 
                                transforms.ToTensor(),
                                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
                                ])

      
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


train = ReplicaDataset(config.train_dir, transform)
test = ReplicaDataset(config.test_dir, transform)

test_abs = int(len(train) * 0.8)
    # 将训练数据划分为训练集（80%）和验证集（20%）
train_subset, val_subset = random_split(
    train, [test_abs, len(train) - test_abs])

train_loader = DataLoader(
    train_subset,
    batch_size=config.BATCH_SIZE,     
    shuffle=True,
    num_workers=8)
valid_loader = DataLoader(
    val_subset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=8)

test_loader = DataLoader(
    test, 
    batch_size=config.BATCH_SIZE, 
    num_workers=8)
                 
model = Bev(1)

criterion = nn.BCEWithLogitsLoss()

model.to(device)
criterion.to(device)

optimizer = optim.Adam(model.parameters(), lr=config.LR)                       
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) 

train_curve = list()
valid_curve = list()

iter_count = 0  

# 构建 SummaryWriter
writer = SummaryWriter(config.log_dir)  

# image1, image2, labels = image1.to(device), image2.to(device), labels.to(device)

# writer.add_graph(model, (image1, image2))

for epoch in range(config.MAX_EPOCH):

    loss_mean = 0.
    correct = 0.
    total = 0.

    model.train()
    # 遍历 train_loader 取数据
    for i, data in enumerate(train_loader):
        iter_count += 1

        # forward
        # img: (b, 3, 480, 640)
        # label:(b, 2, 240, 320),two class 
        img1, img2, labels = data
        img1 = torch.autograd.Variable(img1)
        img2 = torch.autograd.Variable(img2)
        labels = torch.autograd.Variable(labels)
        
        img1 = img1.to(device)
        img2 = img2.to(device)
        labels = labels.to(device)
        

        grid = torchvision.utils.make_grid(img1*0.2+0.5)
        writer.add_image('image1', grid, iter_count)

        grid1 = torchvision.utils.make_grid(img2*0.2+0.5)
        writer.add_image('image2',  grid1, iter_count)

        grid2 = torchvision.utils.make_grid(labels)
        writer.add_image('labels', grid2, iter_count)
        
        # outputs:(b, 2, 240, 320)
        outputs = model(img1, img2)
        grid_out = torchvision.utils.make_grid(outputs)
        writer.add_image('output', grid_out, iter_count)

        loss = criterion(outputs, labels)
        # loss = loss.requires_grad_()
        
        # for name, p in model.named_parameters():
        #     if p.grad is not None:
        #         print(name)
        #         print(p.grad)
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        # update weights
        optimizer.step()
        

        # 统计分类情况
        # predicted:(b, 240, 320)
        _, predicted = torch.max(outputs.data, 1)
        total += (labels.size()[0] * labels.size()[2] * labels.size()[3])
        _, label = torch.max(labels, 1)
        correct += (predicted.detach().cpu() == label.detach().cpu()).squeeze().sum().numpy()
        
        iou = iou_pytorch(predicted, label)
        
        train_curve.append(loss.item())
        loss_mean += loss.item()
        
        if (i+1) % config.log_interval == 0:
            loss_mean = loss_mean / config.log_interval
            print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%} IOU:{:.2%}".format(
                epoch, config.MAX_EPOCH, i+1, len(train_loader), loss_mean, correct / total, iou))
            loss_mean = 0.

        # 记录数据，保存于event file
        writer.add_scalars("Loss", {"Train": loss.item()}, iter_count)
        writer.add_scalars("Accuracy", {"Train": correct / total}, iter_count)
        writer.add_scalars("IoU", {"Train": iou}, iter_count)
    
        

    # 每个epoch，记录梯度，权值
    for name, param in model.named_parameters():
        writer.add_histogram(name + '_grad', param.grad, epoch)
        writer.add_histogram(name + '_data', param, epoch)

    # scheduler.step()  # 更新学习率
    # 每个 epoch 计算验证集得准确率和loss
    # validate the model
    if (epoch+1) % config.val_interval == 0:

        correct_val = 0.
        total_val = 0.
        loss_val = 0.
        model.eval()
        with torch.no_grad():
            for j, data in enumerate(valid_loader):
                img1, img2, labels = data
                img1 = img1.to(device)
                img2 = img2.to(device)
                labels = labels.to(device)
            
                outputs = model(img1, img2)
                loss_ = criterion(outputs, labels)
                loss_val += loss_.item()

                _, predicted_val = torch.max(outputs.data, 1)
                total_val += (labels.size()[0] * labels.size()[2] * labels.size()[3])
                _, label = torch.max(labels, 1)
                correct_val += (predicted_val.detach().cpu() == label.detach().cpu()).squeeze().sum().numpy()
                iou_val = iou_pytorch(predicted_val, label)
                
                if not os.path.isdir(config.output_dir):
                    os.makedirs(config.output_dir)
                if (epoch + 1) % 5 == 0:
                    write_label = rearrange(label.cpu(), 'b h w -> h (b w)')
                    write_predict = rearrange(predicted_val.cpu(), 'b h w -> h (b w)')
                    write_image = torch.cat((write_label, write_predict), dim=0)
                    cv.imwrite(config.output_dir + f'epoch{epoch}_batch{j}_pred.png', write_image.numpy()*255)

            valid_curve.append(loss_val/valid_loader.__len__())
            print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%} IOU:{:.2%}".format(
                epoch, config.MAX_EPOCH, j+1, len(valid_loader), loss_val, correct_val / total_val, iou_val))

            # 记录数据，保存于event file
            writer.add_scalars("Loss", {"Valid": np.mean(valid_curve)}, iter_count)
            writer.add_scalars("Accuracy", {"Valid": correct_val / total_val}, iter_count)
            writer.add_scalars("IoU", {"Valid": iou_val}, iter_count)
            
                
torch.save(model.state_dict(), config.cp_dir)
            
            

train_x = range(len(train_curve))
train_y = train_curve

train_iters = len(train_loader)
valid_x = np.arange(1, len(valid_curve)+1) * train_iters*config.val_interval # 由于valid中记录的是epochloss，需要对记录点进行转换到iterations
valid_y = valid_curve

plt.plot(train_x, train_y, label='Train')
plt.plot(valid_x, valid_y, label='Valid')

plt.legend(loc='upper right')
plt.ylabel('loss value')
plt.xlabel('Iteration')
plt.savefig('bev_curve.jpg')
plt.show()

