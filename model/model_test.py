import cv2 as cv
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from einops import rearrange
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, ConfusionMatrix, JaccardIndex

from bev import Bev
from config import Config
from data_function import ReplicaDataset
from metrics import iou_pytorch

config = Config()
    
transform = transforms.Compose([ 
                                transforms.ToTensor(),
                                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
                                ])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


test = ReplicaDataset(config.test_dir, transform)
test_loader = DataLoader(
    test, 
    batch_size=config.BATCH_SIZE, 
    num_workers=8)

model = Bev(config.N)

model.load_state_dict(torch.load(config.cp_dir))

# Confmat = ConfusionMatrix(num_classes=2)
# Acc = Accuracy()
# IoU = JaccardIndex(num_classes=2)


criterion = nn.BCEWithLogitsLoss()
model.to(device)
criterion.to(device)


correct = 0.
total = 0.
loss = 0.
model.eval()
for i, data in enumerate(test_loader):

    img1, img2, labels = data
    
    img1 = img1.to(device)
    img2 = img2.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        outputs = model(img1, img2)
        loss_ = criterion(outputs, labels)
        loss += loss_.item()
    # cm = Confmat(outputs, labels)
    # acc = Acc(outputs, labels)
    # iou = IoU(outputs, labels)
    
    _, predicted = torch.max(outputs.data, 1)
    total += (labels.size()[0] * labels.size()[2] * labels.size()[3])
    _, label = torch.max(labels, 1)
    correct += (predicted.detach().cpu() == label.detach().cpu()).squeeze().sum().numpy()
    iou_test = iou_pytorch(predicted, label)
        

    write_label = rearrange(label.cpu(), 'b h w -> h (b w)')
    write_predict = rearrange(predicted.cpu(), 'b h w -> h (b w)')
    write_image = torch.cat((write_label, write_predict), dim=0)
    cv.imwrite(f'test_batch{i}_pred.png', write_image.numpy()*255)
    # print("Test:\t Iteration[{:0>3}/{:0>3}] ConfuseMatirx:{}".format(i, len(test_loader)))
    print("Test:\t Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%} IOU:{:.2%}".format(i, len(test_loader), loss, correct / total, iou_test))
    # print("Test:\t Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc_api:{:.2%} IOU_api:{:.2%}".format(loss, acc, iou))
    print("-----------next-----------------")
