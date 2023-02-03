from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from inceptionv4 import *
from inceptionresnetv2 import *
from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

use_cuda=True
# model_map_fp = {
#     'inceptionv3': 'inception_v3_google-1a9a5a14.pth',
# }
# name='inceptionv3'
# model_dir_fp = 'model'
# model_map = model_map_fp
# model_dir = os.path.join(model_dir_fp, model_map[name])
model = models.inception_v3()
model.load_state_dict(torch.load('model/inception_v3_google-0cc3c7bd.pth'))

model_b1 = inceptionv4(num_classes=1000, pretrained='imagenet')
model_b2 = inceptionresnetv2(num_classes=1000, pretrained='imagenet')

model_b3 = models.wide_resnet101_2()
model_b3.load_state_dict(torch.load('model/wide_resnet101_2-32ee1156.pth'))

img_dir = "data/val_clean"

class MyDataSet(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
            csv_file: 标签文件的路径.
            root_dir: 所有图片的路径.
            transform: 一系列transform操作
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)  # 返回数据集长度

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir,
                                self.data_frame.iloc[idx, 0])  # 获取图片所在路径
        img = Image.open(img_path).convert('RGB')  # 防止有些图片是RGBA格式

        lable = self.data_frame.iloc[idx, 1]-1  # 获取图片的类别标签

        if self.transform:
            img = self.transform(img)

        return img ,lable # 返回图片和标签

# 将图像调整为224×224尺寸并归一化
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
img_augs = transforms.Compose([
    transforms.Resize(330),
    transforms.CenterCrop(size=299),
    transforms.ToTensor(),
    # transforms.Normalize(mean, std)
])
#读取数据集
img_set = MyDataSet(csv_file='data/val_rs.csv',root_dir=img_dir,transform=img_augs)
img_iter = torch.utils.data.DataLoader(img_set, batch_size=1, shuffle=False)

# import random
# from matplotlib import pyplot as plt
# # def denorm(img):
# #     for i in range(img.shape[0]):
# #         img[i] = img[i] * std[i] + mean[i]
# #     return img
# plt.figure(figsize=(8, 8))
# for i in range(9):
#     img, label = img_set[random.randint(0, len(img_set))]
#     # img = denorm(img)
#     img = img.permute(1, 2, 0)
#     ax = plt.subplot(3, 3, i + 1)
#     ax.imshow(img.numpy())
#     ax.set_title("label = %d" % label)
#     ax.set_xticks([])
#     ax.set_yticks([])
# plt.show()

# 定义我们正在使用的设备
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

for data, lable in img_iter:
    # 把数据和标签发送到设备
    data, lable = data.to(device), lable.to(device)

# 初始化网络
model = model.to(device)
model_b1 = model_b1.to(device)
model_b2 = model_b2.to(device)
model_b3 = model_b3.to(device)

# 在评估模式下设置模型。在这种情况下，这适用于Dropout图层
model.eval()
model_b1.eval()
model_b2.eval()
model_b3.eval()

epsilons = [16]

def fgsm_attack(image, epsilon, data_grad):
    # 收集数据梯度的元素符号
    sign_data_grad = data_grad.sign()
    # 通过调整输入图像的每个像素来创建扰动图像
    perturbed_image = image + epsilon*sign_data_grad
    # 添加剪切以维持[0,1]范围
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # 返回被扰动的图像
    return perturbed_image

def acc(test_loader,model):
    # 精度计数器
    correct = 0

    # 循环遍历测试集中的所有示例
    for data, target in test_loader:

        # 把数据和标签发送到设备
        data, target = data.to(device), target.to(device)

        # 设置张量的requires_grad属性，这对于攻击很关键
        data.requires_grad = True

        # 通过模型前向传递数据
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

        if init_pred.item() == target.item():
            correct += 1

    # 计算这个epsilon的最终准确度
    final_acc = correct / float(len(test_loader))
    print("攻击前：Test Accuracy = {} / {} = {}".format( correct, len(test_loader), final_acc))
    # 返回准确性和对抗性示例
    return final_acc

# acc(img_iter,model)
# acc(img_iter,model_b1)
# acc(img_iter,model_b2)
# acc(img_iter,model_b3)

def test( model, device, test_loader, epsilon ,model_b1,model_b2,model_b3):

    # 精度计数器
    correct = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0
    nc = 0
    a=0.9
    b=0.999
    for j in range(5):
        nc=nc+((1-b**(j+1))**0.5)/(1-a**(j+1))#可以放在外面加快速度
    adv_examples = []

    # 循环遍历测试集中的所有示例
    for data, target in test_loader:
        # 把数据和标签发送到设备
        data, target = data.to(device), target.to(device)
        data_raw=data
        # 设置张量的requires_grad属性，这对于攻击很关键
        data.requires_grad = True
        # 通过模型前向传递数据
        output = model(data)

        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        # 如果初始预测是错误的，不打断攻击，继续
        if init_pred.item() != target.item():
            continue
        # 计算损失
        loss = F.nll_loss(output, target)
        # 将所有现有的渐变归零
        model.zero_grad()
        # 计算后向传递模型的梯度
        loss.backward()
        # 收集datagrad
        data_grad = data.grad.data
        # 唤醒攻击
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        # 重新分类受扰乱的图像
        output = model(perturbed_data)
        output_b1 = model_b1(perturbed_data)
        output_b2 = model_b2(perturbed_data)
        output_b3 = model_b3(perturbed_data)

        # 检查是否成功
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # 保存0 epsilon示例的特例
            if (epsilon == 0) and (len(adv_examples) < 10):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                data_raw = data_raw.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), data_raw, adv_ex))
        else:
            # 稍后保存一些用于可视化的示例
            if len(adv_examples) < 10:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                data_raw = data_raw.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), data_raw, adv_ex))
        final_pred1 = output_b1.max(1, keepdim=True)[1]  # get the index of the max log-probability
        final_pred2 = output_b2.max(1, keepdim=True)[1]  # get the index of the max log-probability
        final_pred3 = output_b3.max(1, keepdim=True)[1]  # get the index of the max log-probability

        if final_pred1.item() == target.item():
            correct1 += 1
        if final_pred2.item() == target.item():
            correct2 += 1
        if final_pred3.item() == target.item():
            correct3 += 1

    # 计算这个epsilon的最终准确度
    final_acc = correct/float(len(test_loader))
    final_acc1 = correct1 / float(len(test_loader))
    final_acc2 = correct2 / float(len(test_loader))
    final_acc3 = correct3 / float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy  = {}\tTest Accuracy1  = {}\tTest Accuracy2  = {}\tTest Accuracy3  = {}".format(epsilon, final_acc, final_acc1,final_acc2,final_acc3))

    # 返回准确性和对抗性示例
    return final_acc,final_acc1 ,final_acc2,final_acc3 ,adv_examples

accuracies = []
accuracies1 = []
accuracies2 = []
accuracies3 = []

examples = []

# 对每个epsilon运行测试
for eps in epsilons:
    eps=eps/255
    acc,acc1,acc2,acc3, ex = test(model, device, img_iter, eps,model_b1,model_b2,model_b3)
    accuracies.append(acc)
    accuracies1.append(acc1)
    accuracies2.append(acc2)
    accuracies3.append(acc3)
    examples.append(ex)

plt.figure(figsize=(8, 5))
plt.plot(epsilons, accuracies, "*-",color='blue')
plt.plot(epsilons, accuracies1, "*-",color='green')
plt.plot(epsilons, accuracies2, "*-",color='red')
plt.plot(epsilons, accuracies3, "*-",color='black')

plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, 40, step=8))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.show()

cnt = 0
plt.figure(figsize=(60, 10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons),len(examples[0]) * 2,cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig,adv,orig_img, ex = examples[i][j]
        # plt.title("{} -> {}".format(orig, adv))
        plt.title("original: {}".format(orig))
        orig_img = np.transpose(orig_img, (1, 2, 0))
        plt.imshow(orig_img)
        cnt += 1
        plt.subplot(len(epsilons),len(examples[0]) * 2,cnt)
        plt.title("adversarial: {}".format(adv))
        ex = np.transpose(ex, (1, 2, 0))
        plt.imshow(ex)
plt.tight_layout()
plt.show()

# # 在每个epsilon上绘制几个对抗样本的例子
# cnt = 0
# plt.figure(figsize=(8,10))
# for i in range(len(epsilons)):
#     for j in range(len(examples[i])):
#         orig,adv,ex = examples[i][j]
#         ex=torch.tensor(ex)
#         # img = denorm(img)
#         ex = ex.permute(1, 2, 0)
#         ax = plt.subplot(1, 5, j + 1)
#         ax.imshow(ex.numpy())
#         ax.set_title("{} -> {}".format(orig, adv))
#         ax.set_xticks([])
#         ax.set_yticks([])
# plt.tight_layout()
# plt.show()
