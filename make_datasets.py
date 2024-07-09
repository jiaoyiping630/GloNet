import os
import numpy as np
import cv2
from tqdm import trange
import shutil

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

def class2savepath(c):
    if c == 1:
        class_name = '/GS/'
        rgb_name = '/exam_1/'
    elif c == 2:
        class_name = '/FC/'
        rgb_name = '/exam_2/'
    elif c == 3:
        class_name = '/CC/'
        rgb_name = '/exam_3/'
    elif c == 4:
        class_name = '/OG/'
        rgb_name = '/exam_4/'
    elif c == 5:
        class_name = '/IG/'
        rgb_name = '/exam_5/'
    else:
        print('wrong class:  ' + str(c))
        class_name = 'wrong'
        rgb_name = 'wrong'

    return class_name, rgb_name


# root = '/media/zjk/娱乐/IgA-GLH/pytorch-CycleGAN-and-pix2pix-master/middle20x/test/Masks'
# image_root = '/media/zjk/娱乐/IgA-GLH/pytorch-CycleGAN-and-pix2pix-master/middle20x/test/Images/a/'
# rgb_root = '/media/zjk/娱乐/IgA-GLH/pytorch-CycleGAN-and-pix2pix-master/middle20x/test/Rgbmasks/a/'
# save_path = './test_set/'
#
# image_paths, image_names = file_list(root, '.png')
#
# new_size = 384
#
# for i in trange(len(image_names)):
#     mask = cv2.imread(image_paths[i])
#     mask = np.array(mask[:, :, 0])
#     c = mask[256, 256]
#     if c == 0:
#         print(image_names[i])
#         c = int(input('reset class:'))
#     patch = cv2.imread(image_root + image_names[i] + '.png')
#     patch = np.array(patch)
#     patch = cv2.resize(patch, dsize=(new_size, new_size), interpolation=cv2.INTER_AREA)
#     rgb_mask = cv2.imread(rgb_root + image_names[i] + '.png')
#     class_name, rgb_name = class2savepath(save_path, c)
#     cv2.imwrite(class_name + image_names[i] + '.png', patch)
#     cv2.imwrite(rgb_name + image_names[i] + '.png', rgb_mask)

root = '/media/zjk/娱乐/ZhongdaG/zhongda/10-23/Masks'
image_root = '/media/zjk/娱乐/ZhongdaG/zhongda/10-23/Images/'
rgb_root = '/media/zjk/娱乐/ZhongdaG/zhongda/10-23/Rgbmasks/'

image_paths, image_names = file_list(root, '.png')
new_size = 384

for i in trange(len(image_names)):
    mask = cv2.imread(image_paths[i])
    mask = np.array(mask[:, :, 0])
    c = mask[256, 256]
    if c == 0:
        print(image_names[i])
        c = int(input('reset class:'))
    patch = cv2.imread(image_root + image_names[i] + '.png')
    patch = np.array(patch)
    patch = cv2.resize(patch, dsize=(new_size, new_size), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, dsize=(new_size, new_size), interpolation=cv2.INTER_NEAREST)
    rgb_mask = cv2.imread(rgb_root + image_names[i] + '.png')
    class_name, rgb_name = class2savepath(c)
    patch_id = int(image_names[i].split('_')[0])
    if (patch_id % 5) == 100 or (patch_id % 5) == 101:
        save_path = './datasets/10-23-new/Images/test' + class_name + image_names[i] + '.png'
        save_rgb_path = './datasets/10-23-new/exam/test' + rgb_name + image_names[i] + '.png'
        save_mask_path = './datasets/10-23-new/Masks/test' + class_name + image_names[i] + '.png'
        cv2.imwrite(save_path, patch)
        cv2.imwrite(save_rgb_path, rgb_mask)
        cv2.imwrite(save_mask_path, mask)
    else:
        save_path = './datasets/10-23-new/Images/test_whole' + class_name + image_names[i] + '.png'
        save_rgb_path = './datasets/10-23-new/exam/test_whole' + rgb_name + image_names[i] + '.png'
        save_mask_path = './datasets/10-23-new/Masks/test_whole' + class_name + image_names[i] + '.png'
        cv2.imwrite(save_path, patch)
        cv2.imwrite(save_rgb_path, rgb_mask)
        cv2.imwrite(save_mask_path, mask)
