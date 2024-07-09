from PIL import Image
import os
import numpy as np

def file_list(class_list, root_path, format):
    path_WSI_imgs = []  # 图片的绝对路径
    names = []
    for l in range(len(class_list)):
        path = root_path + class_list[l]
        path_WSI = os.walk(path)  # 目录下的东西
        image_WSI_paths = []  # 图片的小路径
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

root = './datasets/Images/train'
class_list = ['/beijing',
              '/pizhi',
              '/suizhi']
# class_list = ['/Crescent',
#               '/Mesangial_hyperplasiac',
#               '/Normal_glomeruli',
#               '/Sclerosis',
#               '/Segmental_sclerosis']
image_path, names = file_list(class_list, root, '.png')
image_size = 384
num = 1
final_mean1 = 0
final_mean2 = 0
final_mean3 = 0
max1 = 0
max2 = 0
max3 = 0
min1 = 2
min2 = 2
min3 = 2
sum1 = 0
sum2 = 0
sum3 = 0
for i in range(len(image_path)):
    image = Image.open(image_path[i])
    image = np.array(image)
    ch1 = image[:, :, 0] / 255
    ch2 = image[:, :, 1] / 255
    ch3 = image[:, :, 2] / 255
    mean1 = np.sum(ch1) /(image_size * image_size)
    mean2 = np.sum(ch2) /(image_size * image_size)
    mean3 = np.sum(ch3) /(image_size * image_size)
    maxi_1 = np.max(ch1)
    maxi_2 = np.max(ch2)
    maxi_3 = np.max(ch3)
    max1 = max(maxi_1, max1)
    max2 = max(maxi_2, max2)
    max3 = max(maxi_3, max3)
    mini_1 = np.min(ch1)
    mini_2 = np.min(ch2)
    mini_3 = np.min(ch3)
    min1 = min(mini_1, min1)
    min2 = min(mini_2, min2)
    min3 = min(mini_3, min3)
    final_mean1 = (final_mean1 * num + mean1) / (num + 1)
    final_mean2 = (final_mean2 * num + mean2) / (num + 1)
    final_mean3 = (final_mean3 * num + mean3) / (num + 1)
    num = num + 1
    if num % 100 ==0:
        print(num)
num = 0
for m in range(len(image_path)):
    image = Image.open(image_path[i])
    image = np.array(image)
    ch1 = (image[:, :, 0] / 255) - final_mean1
    ch2 = (image[:, :, 1] / 255) - final_mean2
    ch3 = (image[:, :, 2] / 255) - final_mean3
    sum1 = sum1 + np.sum(np.square(ch1))
    sum2 = sum2 + np.sum(np.square(ch2))
    sum3 = sum3 + np.sum(np.square(ch3))
    if num % 100 ==0:
        print(num)
    num = num + 1
std1 = np.sqrt(sum1 / (len(image_path) * image_size * image_size))
std2 = np.sqrt(sum2 / (len(image_path) * image_size * image_size))
std3 = np.sqrt(sum3 / (len(image_path) * image_size * image_size))

print('channel1 :')
print('Mean:' + str(final_mean1) +'  Max:' + str(max1) + '  Min:' + str(min1) + '  Std:' + str(std1))
print('channel2 :')
print('Mean:' + str(final_mean2) +'  Max:' + str(max2) + '  Min:' + str(min2) + '  Std:' + str(std2))
print('channel3 :')
print('Mean:' + str(final_mean3) +'  Max:' + str(max3) + '  Min:' + str(min3) + '  Std:' + str(std3))
print('Rooot:')
print(root)