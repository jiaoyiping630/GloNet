import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm
import torch.nn.functional as f

import matplotlib.pyplot as plt


def read_split_data(root: str, val_rate: float = 0.01):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list

def focla_loss(predict, label, num_class):
    one_hot_label = f.one_hot(label, num_class)
    ce_loss = -torch.sum((torch.log(predict) * one_hot_label), dim=1)
    focal = torch.sum(predict * one_hot_label, dim=1)
    #focal = (1 - focal) / (4 * focal + (1 - focal))
    focal = 0.5 * (focal ** 2)
    loss = torch.mean(focal * ce_loss)
    return loss

def jk_loss(predict, label, num_class):
    one_hot_label = f.one_hot(label, num_class)
    ce_loss = -torch.sum((torch.log(predict) * one_hot_label), dim=1)
    focal_gt = torch.sum(predict * one_hot_label, dim=1)
    focal_gt = (4 * focal_gt) / (4 * focal_gt + (1 - focal_gt))
    pre_other = (predict - (predict * one_hot_label))
    label_other = torch.argmax(pre_other, dim=1)#.data
    one_hot_other = f.one_hot(label_other, num_class)
    focal_other = torch.sum(predict * one_hot_other, dim=1)
    focal_other = (4 * focal_other) / (4 * focal_other + (1 - focal_other))
    focal = focal_other / (focal_gt + focal_other)
    focal = 0.5 * (focal ** 2)
    loss = torch.mean(focal * ce_loss)
    return loss

def multi_class_focal(predict, label, num_class):
    one_hot_label = f.one_hot(label, num_class)
    ce_loss = -torch.sum((torch.log(predict) * one_hot_label), dim=1)
    focal_gt = torch.sum(predict * one_hot_label, dim=1)
    pre_other = (predict - (predict * one_hot_label))
    label_other = torch.argmax(pre_other, dim=1)#.data
    one_hot_other = f.one_hot(label_other, num_class)
    focal_max = torch.sum(predict * one_hot_other, dim=1)
    focal = focal_max / (focal_gt + focal_max)
    focal = (focal ** 2)
    loss = torch.mean(focal * ce_loss)
    return loss

def dice_loss(predict, label, num_classes):
    smooth = 1e-6
    if num_classes > 1:
        one_hot_label = f.one_hot(label, num_classes + 1)
        max_dice = 0
        for c in range(num_classes):
            i = torch.sum(one_hot_label[:, :, :, c + 1] * predict[:, c + 1, :, :])
            u = torch.sum(one_hot_label[:, :, :, c + 1] + predict[:, c + 1, :, :])
            dice = (2 * i + smooth) / (u + smooth)
            if dice > max_dice:
                max_dice = dice
        loss = 1 - max_dice
    else:
        i = torch.sum(predict * label)
        u = torch.sum(predict + label)
        dice = (2 * i + smooth) / (u + smooth)
        loss = 1 - dice

    return loss


def train_one_epoch(model, optimizer, data_loader, device, epoch, num_class):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    #average_loss = torch.zeros((num_class)).to(device)
    average_num = 0
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pre_cla = model(images.to(device))
        pred_classes = torch.max(pre_cla, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss_cla = multi_class_focal(pre_cla, labels.to(device), num_class=num_class)
        #average_loss += torch.mean(pre_cla, dim=0)
        average_num += 1
        loss = loss_cla
        loss.backward()

        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
    #print(average_loss / average_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        mask = mask.to(device)
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        # data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
        #                                                                        accu_loss.item() / (step + 1),
        #
        #                                                                        accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
