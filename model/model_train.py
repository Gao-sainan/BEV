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
from loss_function import DiceLoss
import torchvision
import cv2 as cv
import os


config = Config()
    
transform = transforms.Compose([ 
                                transforms.ToTensor(),
                                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
                                ]
)
      
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


train = ReplicaDataset(config.train_dir, transform)
valid = ReplicaDataset(config.valid_dir, transform)

train_loader = DataLoader(train, batch_size=config.BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid, batch_size=config.BATCH_SIZE)
                 
model = Bev(cls=config.cls)

criterion = nn.BCEWithLogitsLoss()
dice = DiceLoss()

model.to(device)
criterion.to(device)


optimizer = optim.SGD(model.parameters(), lr=config.LR, momentum=0.9)                       
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  

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

        # forward
        # img: (b, 3, 480, 640)
        # label:(b, 2, 240, 320),two class 
        img1, img2, labels = data
        
        img1 = img1.to(device)
        img2 = img2.to(device)
        labels = labels.to(device)

        grid = torchvision.utils.make_grid(img1)
        writer.add_image('image1', grid, 0)

        grid1 = torchvision.utils.make_grid(img2)
        writer.add_image('image2', grid1, 1)

        grid2 = torchvision.utils.make_grid(labels)
        writer.add_image('labels', grid2, 2)
        
        # outputs:(b, 1, 240, 320)
        outputs = model(img1, img2)
        grid_out = torchvision.utils.make_grid(outputs)
        writer.add_image('output', grid_out, 3)

        # backward
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        # for p in model.parameters():
        #     if p.grad is not None:
        #         print(p.grad.data)
        loss.backward()

        # update weights
        optimizer.step()
        

        # 统计分类情况
        # predicted:(b, 240, 320)
        _, predicted = torch.max(outputs.data, 1)
            
        
        # total += (labels.size(0) * labels.size(2) * labels.size(3))
        # _, labels = torch.max(labels, 1)
        # correct += (predicted == labels).squeeze().sum().numpy()
        dice_loss = dice(outputs, labels)
        
        train_curve.append(loss.item())
        loss_mean += loss.item()
        
        if (i+1) % config.log_interval == 0:
            loss_mean = loss_mean / config.log_interval
            print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Dice:{:.2%}".format(
                epoch, config.MAX_EPOCH, i+1, len(train_loader), loss_mean, dice_loss))
            loss_mean = 0.

        # 记录数据，保存于event file
        writer.add_scalars("Loss", {"Train": loss.item()}, iter_count)
        writer.add_scalars("Accuracy", {"Train": dice_loss}, iter_count)

    # 每个epoch，记录梯度，权值
    for name, param in model.named_parameters():
        writer.add_histogram(name + '_grad', param.grad, epoch)
        writer.add_histogram(name + '_data', param, epoch)

    scheduler.step()  # 更新学习率
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

                _, predicted_val = torch.max(outputs.data, 1)
                predicted_val = predicted_val.cpu().detach().numpy()
                for b in range(predicted_val.shape[0]):
                    out_image = predicted_val[b]
                    # out_image = out_image * 255
                    if not os.path.isdir(config.output_dir):
                        os.makedirs(config.output_dir)
                    plt.imshow(out_image, cmap='gray')
                    plt.show()
                    plt.imsave(config.output_dir + f'epoch{epoch}_batch{j}_no{b}_out.png', out_image, cmap='gray')
                # total_val += (labels.size(0) * labels.size(2) * labels.size(3))
                # _, labels = torch.max(labels, 1)
                # correct_val += (predicted == labels).squeeze().sum().numpy()
                dice_loss_val = dice(outputs, labels)

                loss_val += loss_.item()

            valid_curve.append(loss_val/valid_loader.__len__())
            print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Dice:{:.2%}".format(
                epoch, config.MAX_EPOCH, j+1, len(valid_loader), loss_val, dice_loss_val))

            # 记录数据，保存于event file
            writer.add_scalars("Loss", {"Valid": np.mean(valid_curve)}, iter_count)
            writer.add_scalars("Accuracy", {"Valid": dice_loss}, iter_count)

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
plt.show()
