import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class CarsDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.transforms = transform
        image_paths = [i + '/CameraRGB' for i in img_dir]
        seg_paths = [i + '/CameraSeg' for i in img_dir]
        self.images_paths, self.masks_paths = [], []
        self.images, self.masks = [], []
        for i in image_paths:
            imgs = os.listdir(i)
            self.images_paths.extend([i + '/' + img for img in imgs])
        for i in seg_paths:
            masks = os.listdir(i)
            self.masks_paths.extend([i + '/' + mask for mask in masks])
        for i in range(len(self.images_paths)):
            img, mask = self.read_imgs(i)
            self.images.append(img)
            self.masks.append(mask)

    def read_imgs(self, idx):
        img = np.array(Image.open(self.images_paths[idx]))
        mask = np.array(Image.open(self.masks_paths[idx]))
        if self.transforms is not None:
            aug = self.transforms(image=img, mask=mask)
            img = aug['image']
            mask = aug['mask']
            mask = torch.max(mask, dim=2)[0]
        return img, mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        mask = self.masks[index]
        return img, mask


def get_images(image_dir, transform=None, batch_size=1, shuffle=True, pin_memory=True, test_size_=0.2):
    data = CarsDataset(image_dir, transform=transform)
    train_size = int((1 - test_size_) * data.__len__())
    test_size = data.__len__() - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])
    train_batch = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                                              pin_memory=pin_memory)
    test_batch = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle,
                                             pin_memory=pin_memory)
    return train_batch, test_batch
