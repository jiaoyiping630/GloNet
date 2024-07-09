# coding=utf-8
"""
@File   : dataset.py
@Time   : 2020/01/07
@Author : Zengrui Zhao
"""
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from pathlib import Path
import os
from PIL import Image
from albumentations.augmentations.transforms import RandomCrop, Blur, RandomBrightnessContrast, RandomGamma, HueSaturationValue
import matplotlib.pyplot as plt


class Data(Dataset):
    def __init__(self, root=Path(__file__), cropSize=(512, 512), isAugmentation=False, mode='train'):
        self.mode = mode
        self.isAugmentation = isAugmentation
        self.crop = False
        self.cropSize = cropSize
        self.root = root
        self.imgs = os.listdir(Path(root) / 'Images')
        assert(mode in ['train', 'test'])
        self.toTensor = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.6223, 0.4734, 0.7026), (0.1646, 0.1791, 0.1064))])

    def augmentation(self, img):
        blur = Blur()
        hsv = HueSaturationValue()
        gamma = RandomGamma()
        brightnessContrast = RandomBrightnessContrast()
        img = hsv.apply(img)
        img = gamma.apply(img)
        img = brightnessContrast.apply(img)
        img = blur.apply(img)
        return img

    def __getitem__(self, item):
        imgPath = Path(Path(self.root) / 'Images' / self.imgs[item])
        maskPath = Path(Path(self.root) / 'Masks' / (self.imgs[item].split('.')[0] + '.png'))
        if self.mode == 'train':
            img = np.array(Image.open(imgPath).convert('RGB'))
            mask = Image.open(maskPath)
            if self.crop:
                while True:
                    transform = RandomCrop(self.cropSize[0], self.cropSize[1])
                    wh = transform.get_params()
                    img_ = transform.apply(np.array(img), h_start=wh['h_start'], w_start=wh['w_start'])
                    mask_ = transform.apply(np.array(mask), h_start=wh['h_start'], w_start=wh['w_start'])
                    img, mask = img_, mask_
                    break
            if self.isAugmentation:
                img = self.augmentation(img)
            mask = np.int64(np.array(mask))
            return self.toTensor(img), mask[None, ...]
        else:
            img = np.array(Image.open(imgPath).convert('RGB'))
            mask = Image.open(maskPath)
            mask = np.int64(np.array(mask))
            return self.toTensor(img), mask[None, ...]

    def __len__(self):
        return len(self.imgs)

