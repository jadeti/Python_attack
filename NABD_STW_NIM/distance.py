from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from inceptionv4 import *
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

use_cuda=True
model = models.inception_v3()
model.load_state_dict(torch.load('model/inception_v3_google-0cc3c7bd.pth'))
model_b1 = inceptionv4(num_classes=1000, pretrained='imagenet')

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

# 定义我们正在使用的设备
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

for data, lable in img_iter:
    # 把数据和标签发送到设备
    data, lable = data.to(device), lable.to(device)

# 初始化网络
model = model.to(device)
model_b1 = model_b1.to(device)
# 在评估模式下设置模型。在这种情况下，这适用于Dropout图层
model.eval()
model_b1.eval()

epsilons = [16]
import random
def input_diversity(image, low=266, high=299 , div_prob=0.5):
    if random.random() > div_prob:
        return image
    rnd = random.randint(low, high)
    rescaled = F.interpolate(image, size=[rnd, rnd], mode='bilinear')
    h_rem = high - rnd
    w_rem = high - rnd
    pad_top = random.randint(0, h_rem)
    pad_bottom = h_rem - pad_top
    pad_left = random.randint(0, w_rem)
    pad_right = w_rem - pad_left
    padded = F.pad(rescaled, [pad_top, pad_bottom, pad_left, pad_right], 'constant', 0)
    return padded

def input_s(image, div_prob):
    if random.random() > div_prob:
        return image
    padded = image/2
    return padded

def input_x(image, div_prob):
    xuanzhuan = transforms.RandomRotation(30)
    if random.random() > div_prob:
        return image
    padded=xuanzhuan(image)
    return padded

def fgsm_attack(image, epsilon, data_grad):
    # 收集数据梯度的元素符号
    sign_data_grad = data_grad.sign()
    # 通过调整输入图像的每个像素来创建扰动图像
    perturbed_image = image + epsilon*sign_data_grad
    # 添加剪切以维持[0,1]范围
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # 返回被扰动的图像
    return perturbed_image

def i_fgsm_attack(image, epsilon,t,target,model):
    momentum=1
    alpha = epsilon / t  # 每次只改变一小步
    perturbed_data = image
    max_x=image+epsilon
    min_x=image-epsilon
    grads = torch.zeros_like(perturbed_data, requires_grad=False)
    for i in range(t):  # 共迭代 t 次
        perturbed_data.requires_grad = True
        outputs = model(perturbed_data)
        loss = F.cross_entropy(outputs, target)
        loss.backward()
        noise = perturbed_data.grad
        perturbed_image = perturbed_data + alpha * noise.sign()
        perturbed_data = torch.max(torch.min(perturbed_image, max_x), min_x).detach()
        perturbed_data = torch.clamp(perturbed_data, 0, 1.0).detach()
        output = model(perturbed_data)  # 代入扰动后的数据
        final_pred = output.max(1, keepdim=True)[1]  # 预测选项
        if(final_pred.item()!=target.item()):
            break
    return perturbed_data

def mi_fgsm_attack(image, epsilon,t,target,model):
    momentum=1
    alpha = epsilon / t  # 每次只改变一小步
    perturbed_data = image
    max_x=image+epsilon
    min_x=image-epsilon
    grads = torch.zeros_like(perturbed_data, requires_grad=False)
    for i in range(t):  # 共迭代 t 次
        perturbed_data.requires_grad = True
        outputs = model(perturbed_data)
        loss = F.cross_entropy(outputs, target)
        loss.backward()
        new_grad = perturbed_data.grad
        noise = momentum * grads + (new_grad) / torch.norm(new_grad, p=1)
        perturbed_image = perturbed_data + alpha * noise.sign()
        perturbed_data = torch.max(torch.min(perturbed_image, max_x), min_x).detach()
        perturbed_data = torch.clamp(perturbed_data, 0, 1.0).detach()
        grads = noise
        output = model(perturbed_data)  # 代入扰动后的数据
        final_pred = output.max(1, keepdim=True)[1]  # 预测选项
        if(final_pred.item()!=target.item()):
            break
    return perturbed_data

def ni_fgsm_attack(image, epsilon,t,target,model):
    decay_factor=1
    alpha = epsilon / t  # 每次只改变一小步
    perturbed_data = image
    max_x=image+epsilon
    min_x=image-epsilon
    grads = torch.zeros_like(perturbed_data, requires_grad=True)
    for i in range(t):  # 共迭代 t 次
        nes_advdata = perturbed_data + alpha * decay_factor * grads
        nes_advdata.requires_grad_ = True
        nes_advdata.retain_grad()#对非叶节点(即中间节点张量)张量启用用于保存梯度的属性(.grad). 默认情况下对于非叶节点张量是禁用该属性grad,计算完梯度之后就被释放回收内存,不会保存中间结果的梯度.)
        outputs = model(nes_advdata)
        loss = F.cross_entropy(outputs, target)
        loss.backward(retain_graph=True)
        new_grad = nes_advdata.grad
        noise = decay_factor * grads + (new_grad) / torch.norm(new_grad, p=1)#梯度的一范
        perturbed_image = perturbed_data + alpha * noise.sign()
        perturbed_data = torch.max(torch.min(perturbed_image, max_x), min_x).detach()
        perturbed_data = torch.clamp(perturbed_data, 0, 1.0).detach()  # range [0, 1]
        grads = noise
        output = model(perturbed_data)  # 代入扰动后的数据
        final_pred = output.max(1, keepdim=True)[1]  # 预测选项
        if(final_pred.item()!=target.item()):
            break
    return perturbed_data

# 新算法攻击代码
def nabd_stw_attack(image, epsilon,t,target,model,nc):
    decay_factor = 1
    perturbed_data = image
    max_x = image + epsilon
    min_x = image - epsilon
    a=0.9
    b=0.999
    A=0
    B=0
    flip=transforms.RandomHorizontalFlip(p=0.5)
    for i in range(t):  # 共迭代 t 次
        at = epsilon * (((1 - b ** (i + 1)) ** 0.5) / (1 - a ** (i + 1))) / nc
        nes_advdata = perturbed_data
        nes_advdata=nes_advdata.requires_grad_(True)
        nes_advdata.retain_grad()  # 对非叶节点(即中间节点张量)张量启用用于保存梯度的属性(.grad). 默认情况下对于非叶节点张量是禁用该属性grad,计算完梯度之后就被释放回收内存,不会保存中间结果的梯度.)
        outputs_fz = model(flip(nes_advdata))
        loss = F.cross_entropy(outputs_fz, target)
        loss.backward(retain_graph=True)
        noise_fz = nes_advdata.grad  # .detach()
        outputs_py = model(input_diversity(nes_advdata, low=266, high=299, div_prob=0.5))
        loss = F.cross_entropy(outputs_py, target)
        loss.backward(retain_graph=True)
        noise_py = nes_advdata.grad  # .detach()
        outputs_cd = model(input_s(nes_advdata, div_prob=0.5))
        loss = F.cross_entropy(outputs_cd, target)
        loss.backward(retain_graph=True)
        noise_cd = nes_advdata.grad  # .detach()
        outputs_xz = model(input_x(nes_advdata,0.5))
        loss = F.cross_entropy(outputs_xz, target)
        loss.backward(retain_graph=True)
        noise_xz = nes_advdata.grad
        noise = (noise_fz + noise_py +noise_cd+noise_xz) / 4
        noise_ = noise/(1 - a ** (i + 1))#梯度偏差校正
        A = a * A + (1 - a) * noise  # 一阶估计
        B = b * B + (1 - b) * (noise - A) ** 2  # 二阶估计
        A_ = A / (1 - a ** (i + 1))
        B_ = (B+(1e-8))/ (1 - b ** (i + 1))  # 偏差校正
        m_ = a * A_+(1-a) * noise_#引入nesterov
        r = m_ / (B_ ** 0.5 + (1e-8))#引入nifgsm时应该加这个r
        perturbed_image = perturbed_data + at * r.sign()
        perturbed_data = torch.max(torch.min(perturbed_image, max_x), min_x).detach()
        perturbed_data = torch.clamp(perturbed_data, 0, 1.0).detach()  # range [0, 1]
        output = model(perturbed_data)  # 代入扰动后的数据
        final_pred = output.max(1, keepdim=True)[1]  # 预测选项
        if (final_pred.item() != target.item()):
            break
    return perturbed_data

def nabd_stw_nim_attack(image, epsilon,t,target,model,nc):
    decay_factor = 1
    perturbed_data = image
    max_x = image + epsilon
    min_x = image - epsilon
    a=0.9
    b=0.999
    A=0
    B=0
    flip=transforms.RandomHorizontalFlip(p=0.5)
    # hdt=torchvision.transforms.RandomGrayscale(p=0.5)#灰度变换
    grads = torch.zeros_like(perturbed_data, requires_grad=True)
    for i in range(t):  # 共迭代 t 次
        at = epsilon * (((1 - b ** (i + 1)) ** 0.5) / (1 - a ** (i + 1))) / nc
        nes_advdata = perturbed_data + at * decay_factor * grads
        nes_advdata=nes_advdata.requires_grad_(True)
        nes_advdata.retain_grad()  # 对非叶节点(即中间节点张量)张量启用用于保存梯度的属性(.grad). 默认情况下对于非叶节点张量是禁用该属性grad,计算完梯度之后就被释放回收内存,不会保存中间结果的梯度.)
        # outputs = model(input_x(nes_advdata,0.5))
        # loss = F.cross_entropy(outputs, target)
        # # model.zero_grad()
        # loss.backward(retain_graph=True)
        # noise = nes_advdata.grad#.detach()
        outputs_fz = model(flip(nes_advdata))
        loss = F.cross_entropy(outputs_fz, target)
        loss.backward(retain_graph=True)
        noise_fz = nes_advdata.grad  # .detach()
        outputs_py = model(input_diversity(nes_advdata, low=266, high=299, div_prob=0.5))
        loss = F.cross_entropy(outputs_py, target)
        loss.backward(retain_graph=True)
        noise_py = nes_advdata.grad  # .detach()
        outputs_cd = model(input_s(nes_advdata, div_prob=0.5))
        loss = F.cross_entropy(outputs_cd, target)
        loss.backward(retain_graph=True)
        noise_cd = nes_advdata.grad  # .detach()
        outputs_xz = model(input_x(nes_advdata,0.5))
        loss = F.cross_entropy(outputs_xz, target)
        loss.backward(retain_graph=True)
        noise_xz = nes_advdata.grad
        noise = (noise_fz + noise_py +noise_cd+noise_xz) / 4
        noise_ = noise/(1 - a ** (i + 1))#梯度偏差校正
        A = a * A + (1 - a) * noise  # 一阶估计
        B = b * B + (1 - b) * (noise - A) ** 2  # 二阶估计
        A_ = A / (1 - a ** (i + 1))
        B_ = (B+(1e-8))/ (1 - b ** (i + 1))  # 偏差校正
        m_ = a * A_+(1-a) * noise_#引入nesterov
        r = m_ / (B_ ** 0.5 + (1e-8))#引入nifgsm时应该加这个r
        # noise_1 = 299 * 299 * 3 * (r) / torch.norm(r, p=1)
        # perturbed_image = perturbed_data + at * noise_1
        # noise = decay_factor * grads + (noise) / torch.norm(noise, p=1)  # 梯度的一范
        perturbed_image = perturbed_data + at * r.sign()
        # perturbed_image = perturbed_data + at * r.tanh()
        perturbed_data = torch.max(torch.min(perturbed_image, max_x), min_x).detach()
        perturbed_data = torch.clamp(perturbed_data, 0, 1.0).detach()  # range [0, 1]
        grads = r
        output = model(perturbed_data)  # 代入扰动后的数据
        final_pred = output.max(1, keepdim=True)[1]  # 预测选项
        if (final_pred.item() != target.item()):
            break
    return perturbed_data

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

def test( model, device, test_loader, epsilon ,model_b1):

    # 精度计数器
    correct = 0
    correct1 = 0
    dis_sum=0
    num=0
    nc = 0
    a=0.9
    b=0.999
    for j in range(5):
        nc=nc+((1-b**(j+1))**0.5)/(1-a**(j+1))#可以放在外面加快速度
    adv_examples = []

    # 循环遍历测试集中的所有示例
    for data, target in test_loader:
        # 把数据和标签发送到设备
        num+=1
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
        # perturbed_data = fgsm_attack(data, epsilon, data_grad)
        perturbed_data = i_fgsm_attack(data, epsilon,5,target,model)
        # perturbed_data = mi_fgsm_attack(data,  epsilon, 5, target ,model)
        # perturbed_data = ni_fgsm_attack(data, epsilon,5,target,model)
        # perturbed_data = nabd_stw_attack(data,  epsilon, 5, target ,model,nc)
        # perturbed_data = nabd_stw_nim_attack(data,  epsilon, 5, target ,model,nc)

        # 重新分类受扰乱的图像
        output = model(perturbed_data)
        output_b1 = model_b1(perturbed_data)

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
        if final_pred1.item() == target.item():
            correct1 += 1
        # 计算原图像与对抗图像之间的距离
        if num<=200:
            pdist = nn.PairwiseDistance(p=2)  # p=2就是计算欧氏距离，p=1就是曼哈顿距离.
            dis = pdist(data, perturbed_data)  # 计算各自每一行之间的欧式距离。
            # dis = dis.squeeze().detach().cpu().numpy()
            dis_sum += dis

    # 计算这个epsilon的最终准确度
    final_acc = correct/float(len(test_loader))
    final_acc1 = correct1 / float(len(test_loader))

    print("Epsilon: {}\tTest Accuracy  = {}\tTest Accuracy1  = {}".format(epsilon, final_acc, final_acc1))
    print("扰动量：{}".format(dis_sum.sum()))
    # 返回准确性和对抗性示例
    return final_acc, final_acc1,adv_examples

accuracies = []
accuracies1 = []
examples = []

# 对每个epsilon运行测试
for eps in epsilons:
    eps=eps/255
    acc,acc1,ex = test(model, device, img_iter, eps,model_b1)
    accuracies.append(acc)
    accuracies1.append(acc1)
    examples.append(ex)

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
plt.title("NABD-STW-NIM")
plt.tight_layout()
plt.show()

