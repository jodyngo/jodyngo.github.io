import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np
import random
import cv2


########################### Data Augmentation ###########################


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask):
        image = (image - self.mean) / self.std
        mask /= 255
        return image, mask


class RandomCrop(object):
    def __call__(self, image, mask):
        H, W, _ = image.shape
        randw = np.random.randint(W / 8)
        randh = np.random.randint(H / 8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H + offseth - randh, offsetw, W + offsetw - randw
        return image[p0:p1, p2:p3, :], mask[p0:p1, p2:p3]


class RandomFlip(object):
    def __call__(self, image, mask):
        if np.random.randint(2) == 0:
            return image[:, ::-1, :], mask[:, ::-1]
        else:
            return image, mask


class RandomBlur:
    def __init__(self, prob=0.1):
        self.prob = prob

    def __call__(self, im, label, body=None, detail=None):
        if self.prob <= 0:
            n = 0
        elif self.prob >= 1:
            n = 1
        else:
            n = int(1.0 / self.prob)
        if n > 0:
            if np.random.randint(0, n) == 0:
                radius = np.random.randint(3, 10)
                if radius % 2 != 1:
                    radius = radius + 1
                if radius > 9:
                    radius = 9
                im = cv2.GaussianBlur(im, (radius, radius), 0, 0)
        return im, label  # , body, detail


class RandomBrightness(object):
    def __call__(self, image, mask, body=None, detail=None):
        contrast = np.random.rand(1) + 0.5
        light = np.random.randint(-15, 15)
        inp_img = contrast * image + light
        return np.clip(inp_img, 0, 255), mask  # , body, detail


class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask


class Resize_Label(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image


class ToTensor(object):
    def __call__(self, image, mask):
        # image = torch.from_numpy(image)
        # image = image.permute(2, 0, 1)
        # mask = torch.from_numpy(mask)
        image = image.transpose((2, 0, 1))
        image_tensor = torch.from_numpy(image)
        label_tensor = torch.LongTensor(np.array(mask, dtype=np.int))
        return image_tensor, label_tensor


mean = np.array([[[124.55, 118.90, 102.94]]])
std = np.array([[[56.77, 55.97, 57.50]]])


class ImageDataTrain(data.Dataset):
    def __init__(self, data_root, data_list):
        self.sal_root = data_root
        self.sal_source = data_list

        self.randomblur = RandomBlur()
        self.randombright = RandomBrightness()
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()
        self.resize = Resize(224, 224)

        with open(self.sal_source, 'r') as f:
            self.sal_list = [x.strip() for x in f.readlines()]

        self.sal_num = len(self.sal_list)

    def __getitem__(self, item):
        # sal data loading
        im_name = self.sal_list[item % self.sal_num].split()[0]
        gt_name = self.sal_list[item % self.sal_num].split()[1]

        # load image
        image = cv2.imread(os.path.join(self.sal_root, im_name))  # (300, 400, 3)
        image = np.array(image, dtype=np.float32)
        image -= np.array((104.00699, 116.66877, 122.67892))

        # load label
        label = cv2.imread(os.path.join(self.sal_root, gt_name))  # (300, 400, 3)
        mask = np.array(label, dtype=np.float32)

        # image, mask = self.blur(image, mask)
        # image, mask = self.randombright(image, mask)
        image, mask = self.randomcrop(image, mask)
        image, mask = self.randomflip(image, mask)
        image, mask = self.resize(image, mask)  # (352, 352, 3)

        if len(mask.shape) == 3:
           mask = mask[:, :, 0]
        mask = mask / 255.
        mask = mask[np.newaxis, ...]      # convert mask to (1, 352, 352)
        image = image.transpose(2, 0, 1)  # convert image to (3, 352, 352)

        sal_image = torch.Tensor(image.copy())
        sal_label = torch.Tensor(mask.copy())

        sample = {'sal_image': sal_image, 'sal_label': sal_label}

        return sample

    def __len__(self):
        return self.sal_num


class ImageDataTest(data.Dataset):
    def __init__(self, data_root, data_list):
        self.data_root = data_root
        self.data_list = data_list

        self.resize = Resize_Label(224, 224)

        with open(self.data_list, 'r') as f:
            self.sal_list = [x.strip() for x in f.readlines()]

        self.sal_num = len(self.sal_list)

    def __getitem__(self, item):
        # sal data loading
        path = os.path.join(self.data_root, self.sal_list[item])

        # load image
        image = cv2.imread(path)  # HxWx3  (300, 400, 3)
        in_ = np.array(image, dtype=np.float32)
        im_size = tuple(in_.shape[:2])
        in_ -= np.array((104.00699, 116.66877, 122.67892))

        in_ = self.resize(in_)  # (352, 352, 3)
        in_ = in_.transpose((2, 0, 1))  # (3, 352, 352)

        image_ = torch.Tensor(in_)

        return {'image': image_, 'name': self.sal_list[item % self.sal_num], 'size': im_size}

    def __len__(self):
        return self.sal_num


def get_loader(config, mode='train', pin=False):
    shuffle = False
    if mode == 'train':
        shuffle = True
        dataset = ImageDataTrain(config.train_root, config.train_list)  # insert NTT
        data_loader = data.DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=shuffle,
                                      num_workers=config.num_thread, pin_memory=pin)
    else:
        dataset = ImageDataTest(config.test_root, config.test_list)  # insert NTT
        data_loader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=shuffle,
                                      num_workers=config.num_thread, pin_memory=pin)
    return data_loader
