import os
import numpy as np
import cv2
from tqdm import trange
import shutil

move_list = ['218', '219', '220', '221', '222', '223', '253', '254', '255', '256', '257',
             '269', '274', '275', '279', '282', '284', '285']

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

class_list = ['CC', 'FC', 'GS', 'IG', 'OG']

for c in range(len(class_list)):
    root = './datasets/10-23-new/Images/test_whole/' + class_list[c]
    image_paths, image_names = file_list(root, '.png')
    for i in trange(len(image_names)):
        for m in range(len(move_list)):
            if move_list[m] in image_names[i]:
                des = image_paths[i].replace('test_whole', 'GBM_ANCA')
                shutil.move(src=image_paths[i], dst=des)
