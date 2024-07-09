import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import trange
import numpy as np
import cv2

from model import swin_base_patch4_window12_384 as create_model

def file_list(path, format):
    path_WSI = os.walk(path)  # 目录下的东西
    image_WSI_paths = []  # 图片的小路径
    path_WSI_imgs = []  # 图片的绝对路径
    names = []
    for root_, dirs, files in path_WSI:
        for file in files:
            format_img = os.path.splitext(file)[-1]
            if format_img == format:
                name = os.path.splitext(file)[0]
                image_WSI_paths.append(file)
                names.append(name)
    for image_path in image_WSI_paths:
        path_img = os.path.join(path, image_path)
        path_WSI_imgs.append(path_img)
    return (path_WSI_imgs, names)
def label2rgb(cl, rgbmask):
    if cl == 0:
        rgbmask[:, :, 0] = 0
    elif cl == 1:
        rgbmask[:, :, 2] = 0
    elif cl == 2:
        rgbmask[:, :, 1] = 0
    elif cl == 3:
        rgbmask[:, :, 0] = 0
        rgbmask[:, :, 1] = 0
    else:
        rgbmask[:, :, 1] = 0
        rgbmask[:, :, 2] = 0
    return rgbmask

#######color code#########
# "0": "Segmental_sclerosis",   (0, 255, 255)
# "1": "crescent",              (255,255, 0)
# "2": "mesangial_hyperplasia", (255,0,255)
# "3": "normal_glomeruli",      (0,0,255)
# "4": "sclerosis"              255,0,0)



#device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_list = ['./datasets/Images/test/beijing',
             './datasets/Images/test/pizhi',
             './datasets/Images/test/suizhi']
wrong_list = ['']

img_size = 224
data_transform = transforms.Compose(
    [#transforms.Resize(int(img_size * 1.14)),
     #transforms.CenterCrop(img_size),
     transforms.ToTensor(),
     transforms.Normalize([0.7804, 0.6763, 0.7662], [0.1391, 0.1798, 0.1235])])#5classes:0.7615, 0.6425, 0.7754], [0.1162, 0.1611, 0.0987
                                                                               #4classes:[0.7582, 0.6376, 0.7822], [0.1463, 0.2401, 0.1337]
# create model
model = create_model(num_classes=3).to(device)
# load model weights
save_path = './result/classification'
model_weight_path = "./weights/classification/best-108.pth"
model.load_state_dict(torch.load(model_weight_path, map_location=device))
model.eval()
# if os.path.exists(save_path) is False:
#     os.makedirs(save_path)
#     os.makedirs(save_path + '/seg_pre')
#     os.makedirs(save_path + '/compare')


confuse_metrix = np.zeros((3, 3))
claaaass_num = np.zeros((3))
confuse_rate = np.zeros((3, 3))

for c in range(len(test_list)):
    image_paths, image_names = file_list(test_list[c], '.png')
    for i in trange(len(image_names)):
        img = Image.open(image_paths[i])
        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0)
        with torch.no_grad():
            output = torch.squeeze(model(img.to(device))).cpu()
            predict_cla = torch.argmax(output).numpy()
        claaaass_num[c] += 1
        confuse_metrix[c, predict_cla] += 1
    confuse_rate[c, :] = confuse_metrix[c, :] / claaaass_num[c]

print(confuse_metrix)
print(claaaass_num)
print(confuse_rate)