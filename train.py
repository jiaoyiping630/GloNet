import os
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataSet
from model import swin_base_patch4_window12_384 as create_model
from utils import read_split_data, train_one_epoch, evaluate
from test_moudle import test_moudle
import numpy as np
import time

mean = [0.7804, 0.6763, 0.7662]
std = [0.1391, 0.1798, 0.1235]

def save_best_moudle(root, epoch, best_list, model, num_class):
    confuse_rate = test_moudle(model_path=root + "/each_epoch.pth", mean=mean, std=std, num_class=num_class)
    total_acc = 0
    for b in range(num_class):
        if confuse_rate[b, b] > best_list[b]:
            torch.save(model.state_dict(), root + '/best_' + str(b + 1) + '-' + str(epoch) + '.pth')
            best_list[b] = confuse_rate[b, b]
        total_acc += confuse_rate[b, b]
    if total_acc > best_list[num_class]:
        torch.save(model.state_dict(), root + '/best-{}.pth'.format(epoch))
        best_list[num_class] = total_acc
    print_information = ''
    for p in range(num_class + 2):
        if p < num_class:
            print_information = print_information + 'best_' + str(p + 1) + ':' + str(best_list[p]) + '  '
        else :
            print_information = print_information + 'BEST:' + str(best_list[p] / num_class) + '  '

    print(print_information)
    return best_list

pth_save_root = "./weights/classification_2"

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists(pth_save_root) is False:
        os.makedirs(pth_save_root)

    tb_writer = SummaryWriter()

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    img_size = 384
    data_transform = {
        "train": transforms.Compose([#transforms.RandomResizedCrop(img_size),
                                     #transforms.RandomHorizontalFlip(),
                                     #transforms.RandomRotation(90),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean, std)]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])}##5classes:0.7603, 0.6409, 0.7752], [0.1159, 0.1608, 0.0987
                                                                                                               #4classes:0.7653, 0.6446, 0.7822], [0.1405, 0.2344, 0.1338

    # 实例化训练数据集save_path + '
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)

    # best_acc_1 = torch.zeros(1).to(device)
    # best_acc_2 = torch.zeros(1).to(device)
    # best_acc_3 = torch.zeros(1).to(device)
    # best_acc_4 = torch.zeros(1).to(device)
    # best_acc_5 = torch.zeros(1).to(device)
    # best_acc = torch.zeros(1).to(device)
    best_list = np.zeros((args.num_classes + 2))

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                num_class=args.num_classes)

        # validate
        #val_loss, val_acc = evaluate(model=model,
        #                             data_loader=val_loader,
        #                             device=device,
        #                             epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        #tb_writer.add_scalar(tags[2], val_loss, epoch)
        #tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        # if epoch % 10 == 9:
        #     torch.save(model.state_dict(), "./try2_nf/model-{}.pth".format(epoch))
        #
        # if train_acc >= best_acc:
        #     torch.save(model.state_dict(), "./try2_nf/best.pth".format(epoch))
        torch.save(model.state_dict(), pth_save_root + "/each_epoch.pth")
        if epoch > 0:
            best_list = save_best_moudle(pth_save_root, epoch, best_list, model, num_class = args.num_classes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0001)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="./datasets/Images/train")

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='/media/zjk/文档/Kidney/classification/swin_base_patch4_window12_384_22k.pth',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
