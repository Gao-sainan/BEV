from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
from bev import Bev

import torchvision
import torchvision.transforms as transforms
from config import Config
from data_function import ReplicaDataset
from metrics import iou_pytorch
 
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

c = Config()

# 封装数据加载过程，传递全局数据路径，以保证不同实验间共享数据路径
def load_data(train_dir=c.train_dir, test_dir=c.test_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])
    
    trainset = ReplicaDataset(train_dir, transform)
    testset = ReplicaDataset(test_dir, transform)
 
    return trainset, testset

# 封装训练脚本
# config参数用于指定超参数
# cp_dir参数用于存储检查点
# data_dir参数用于指定数据加载和存储路径
def train_bev(config, checkpoint_dir=None, train_dir=None, test_dir=None):
    model = Bev(config["n_transformer"])
    

    # 将模型封装到nn.DataParallel中以支持多GPU并行训练
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.cuda()
        
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)      # 1个超参数
    
        
    # 用于存储检查点
    if checkpoint_dir:
        # 模型的状态、优化器的状态
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
 
    trainset, testset = load_data(train_dir, test_dir)
 
    test_abs = int(len(trainset) * 0.8)
    # 将训练数据划分为训练集（80%）和验证集（20%）
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs])
 
    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),       # 1个超参数
        shuffle=True,
        num_workers=8)
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)
    
    for epoch in range(25):  # loop over the dataset multiple times
        loss_mean = 0.
        correct = 0.
        total = 0.
        
        model.train()
        # 训练循环
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            img1, img2, labels = data
            img1 = torch.autograd.Variable(img1)
            img2 = torch.autograd.Variable(img2)
            labels = torch.autograd.Variable(labels)
            
            img1 = img1.cuda()
            img2 = img2.cuda()
            labels = labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(img1, img2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            _, predicted = torch.max(outputs.data, 1)
            total += (labels.size()[0] * labels.size()[2] * labels.size()[3])
            _, label = torch.max(labels, 1)
            correct += (predicted.detach().cpu() == label.detach().cpu()).squeeze().sum().numpy()
        
            iou = iou_pytorch(predicted, label)
            
            loss_mean += loss.item()

            if (i+1) % c.log_interval == 0:
                loss_mean = loss_mean / c.log_interval
                print("Training:Epoch[{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%} IOU:{:.2%}".format(
                    epoch, i+1, len(trainloader), loss_mean, correct / total, iou))
                loss_mean = 0.
                
        # validate

        correct_val = 0.
        total_val = 0.
        loss_val = 0.
        model.eval()

        for j, data in enumerate(valloader, 0):
            with torch.no_grad():
                img1, img2, labels = data
                img1 = img1.cuda()
                img2 = img2.cuda()
                labels = labels.cuda()
                
                outputs = model(img1, img2)
                loss_ = criterion(outputs, labels)
                loss_val += loss_.item()
                
                _, predicted_val = torch.max(outputs.data, 1)
                total_val += (labels.size()[0] * labels.size()[2] * labels.size()[3])
                _, label = torch.max(labels, 1)
                correct_val += (predicted_val.detach().cpu() == label.detach().cpu()).squeeze().sum().numpy()
                iou_test = iou_pytorch(predicted_val, label)
                
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)
            
        tune.report(loss=(loss_val / len(valloader)), accuracy=correct_val / total_val)
    print("Finished Training")
        
# 测试集精度
def test_accuracy(net, device="cuda:0"):
    trainset, testset = load_data()
 
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2)
 
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
 
    return correct / total

def main(num_samples=10, max_num_epochs=25, gpus_per_trial=2):
    cp_dir = c.cp_dir
    train_dir = c.train_dir
    test_dir = c.test_dir
    # 加载训练数据
    load_data(train_dir, test_dir)
    # 配置超参数搜索空间
    # 每次实验，Ray Tune会随机采样超参数组合，并行训练模型，找到最优参数组合
    config = {
        "n_transformer": tune.choice([1, 3, 4, 6]),
        # 随机分布采样
        "lr": tune.loguniform(1e-4, 1e-1),
        # 从类别型值中随机选择
        "batch_size": tune.choice([2, 4])
    }
    # ASHAScheduler会根据指定标准提前中止坏实验
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    # 在命令行打印实验报告
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    # 执行训练过程
    result = tune.run(
        partial(train_bev, checkpoint_dir=cp_dir, train_dir=train_dir, test_dir=test_dir),
        # 指定训练资源
        resources_per_trial={"gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)
 
    # 找出最佳实验
    best_trial = result.get_best_trial("loss", "min", "last")
    # 打印最佳实验的参数配置
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))
 
    # 打印最优超参数组合对应的模型在测试集上的性能
    best_trained_model = Bev(best_trial.config["n_transformer"])
    best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.cuda()
 
    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)
 
    test_acc = test_accuracy(best_trained_model)
    print("Best trial test set accuracy: {}".format(test_acc))
 
 
if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=25, gpus_per_trial=1)

