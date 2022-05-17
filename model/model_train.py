import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from einops import rearrange
from tabulate import tabulate
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import (Accuracy, ConfusionMatrix, JaccardIndex, Precision,
                          Recall)

from bev import Bev
from config import Config
from data_function import ReplicaDataset



def load_data(config):
    transform = transforms.Compose([ 
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                ])
    train = ReplicaDataset(config.train_dir, transform)
    test = ReplicaDataset(config.test_dir, transform)


    val_abs = int(len(train) * 0.8)
        # 将训练数据划分为训练集（80%）和验证集（20%）
    train_subset, val_subset = random_split(
        train, [val_abs, len(train) - val_abs])

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
        shuffle=True,
        num_workers=8)
    
    return train_loader, valid_loader, test_loader


# train with earlystopping
def train(model, patience, train_loader, valid_loader, config):

    Acc = Accuracy()
    IoU = JaccardIndex(num_classes=2)
    
    es = 0
    best_acc = 0.
    
    train_curve = list()
    valid_curve = list()

    iter_count = 0  

    # 构建 SummaryWriter
    writer = SummaryWriter(config.log_dir)  


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
            
            img1 = img1.to(config.device)
            img2 = img2.to(config.device)
            labels = labels.to(config.device)
            

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
            
            # backward
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            

            # 统计分类情况
            # predicted:(b, 240, 320)
            _, predicted = torch.max(outputs.data, 1)
            _, label = torch.max(labels, 1)
            acc = Acc(outputs.detach().cpu(), label.detach().cpu())
            iou = IoU(outputs.detach().cpu(), label.detach().cpu())
            
            train_curve.append(loss.item())
            loss_mean += loss.item()
            
            if (i+1) % config.log_interval == 0:
                loss_mean = loss_mean / config.log_interval
                print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%} IOU:{:.2%}".format(
                    epoch, config.MAX_EPOCH, i+1, len(train_loader), loss_mean, acc, iou))
                loss_mean = 0.

            # 记录数据，保存于event file
            writer.add_scalars("Loss", {"Train": loss.item()}, iter_count)
            writer.add_scalars("Accuracy", {"Train": acc}, iter_count)
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
                    img1 = img1.to(config.device)
                    img2 = img2.to(config.device)
                    labels = labels.to(config.device)
                
                    outputs = model(img1, img2)
                    loss_ = criterion(outputs, labels)
                    loss_val += loss_.item()

                    _, predicted_val = torch.max(outputs.data, 1)
                    _, label = torch.max(labels, 1)
                    acc_val = Acc(outputs.detach().cpu(), label.detach().cpu())
                    iou_val = IoU(outputs.detach().cpu(), label.detach().cpu())
                    
                    if not os.path.isdir(config.output_dir):
                        os.makedirs(config.output_dir)
                    if (epoch + 1) % 20 == 0:
                        write_label = rearrange(label.cpu(), 'b h w -> h (b w)')
                        write_predict = rearrange(predicted_val.cpu(), 'b h w -> h (b w)')
                        write_image = torch.cat((write_label, write_predict), dim=0)
                        cv.imwrite(config.output_dir + f'epoch{epoch}_batch{j}_pred.png', write_image.numpy()*255)

                valid_curve.append(loss_val/valid_loader.__len__())
                print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%} IOU:{:.2%}".format(
                    epoch, config.MAX_EPOCH, j+1, len(valid_loader), loss_val, acc_val, iou_val))
                
                # 记录数据，保存于event file
                writer.add_scalars("Loss", {"Valid": np.mean(valid_curve)}, iter_count)
                writer.add_scalars("Accuracy", {"Valid": acc_val}, iter_count)
                writer.add_scalars("IoU", {"Valid": iou_val}, iter_count)
                
                if iou_val > best_acc:
                    best_acc = iou_val
                    es = 0
                    torch.save(model.state_dict(), config.cp_dir)
                else:
                    es += 1
                    print("Counter {} of {}".format(es, patience))

                    if es > patience:
                        print("Early stopping with best_iou: ", best_acc.item(), "and iou for this epoch: ", iou_val.item(), "...")
                        break
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(config.cp_dir))      
    train_x = range(len(train_curve))
    train_y = train_curve

    train_iters = len(train_loader)
    valid_x = np.arange(1, len(valid_curve)+1) * train_iters * config.val_interval # 由于valid中记录的是epochloss，需要对记录点进行转换到iterations
    valid_y = valid_curve

    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.legend(loc='upper right')
    plt.ylabel('loss value')
    plt.xlabel('Iteration')
    plt.savefig('bev_curve.jpg')
    plt.show()
              
    return model

# test
def test(model, test_loader, config):
    Confmat = ConfusionMatrix(num_classes=2)
    Acc = Accuracy()
    IoU = JaccardIndex(num_classes=2)
    Pre = Precision(num_classes=2, mdmc_average='samplewise')
    Rec = Recall(num_classes=2, mdmc_average='samplewise')

    mean_iou = 0.
    mean_precision = 0.
    mean_recall = 0.

    model.eval()
    for i, data in enumerate(test_loader):

        img1, img2, labels = data
        
        img1 = img1.to(config.device)
        img2 = img2.to(config.device)
        labels = labels.to(config.device)
        with torch.no_grad():
            outputs = model(img1, img2)
            loss_ = criterion(outputs, labels)
            # loss += loss_.item()

        
        _, predicted = torch.max(outputs.data, 1)
        _, label = torch.max(labels, 1)
        
        cm = Confmat(outputs.detach().cpu(), label.detach().cpu())
        acc = Acc(outputs.detach().cpu(), label.detach().cpu())
        iou_test = IoU(outputs.detach().cpu(), label.detach().cpu())
        precision = Pre(predicted.detach().cpu(), label.detach().cpu())
        recall = Rec(predicted.detach().cpu(), label.detach().cpu())
        mean_iou += iou_test.item()
        mean_precision +=  precision.item()
        mean_recall += recall.item()
            
        write_label = rearrange(label.cpu(), 'b h w -> h (b w)')
        write_predict = rearrange(predicted.cpu(), 'b h w -> h (b w)')
        write_image = torch.cat((write_label, write_predict), dim=0)
        cv.imwrite(f'test_predict/test_batch{i}_pred.png', write_image.numpy()*255)
        print("Test:\t Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%} IOU:{:.2%} Precision:{:.2%} Recall:{:.2%}".format(i, len(test_loader), loss_, acc, iou_test, precision, recall))

    table = [['test', mean_iou / (i + 1), mean_precision / (i + 1), mean_precision / (i + 1)]]   
    print(tabulate(table, headers=[' ', 'IoU', 'Precison', 'Recall']))



config = Config()
train_loader, valid_loader, test_loader = load_data(config)
model = Bev(config.N)
criterion = nn.BCEWithLogitsLoss()
model.to(config.device)
criterion.to(config.device)

optimizer = optim.Adam(model.parameters(), lr=config.LR,weight_decay=0.0005)                       
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) 
train_model = train(model, 20, train_loader, valid_loader, config)
print('----------------------------------Finish Training!----------------------------------')
test(train_model, test_loader, config)


    
            

