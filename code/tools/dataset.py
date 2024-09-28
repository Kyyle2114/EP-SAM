import os
import numpy as np 
from PIL import Image
from typing import Type

import torch
from torch.utils.data import Dataset
import torchvision.transforms as tr 
import cv2

class ClassifierDataset(Dataset):
    def __init__(self, image_dir, transform):
        """
        Make pytorch Dataset for given task.

        Args:
            image_dir (str): dataset directory
            transform (torchvision.transforms) pytorch image transforms  
        """
        self.image_dir = image_dir
        self.data_images = os.listdir(image_dir)
        self.labels = []
        
        for file in self.data_images:
            if file.endswith('.png'):
                label = int(file[:-4].split('_')[-1])
                self.labels.append(label)
                
        self.transform = transform

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.data_images[idx])
        image = Image.open(img_path)
        label = self.labels[idx]

        image = self.transform(image)
        label = torch.tensor(label)

        return image, label
    
    def __len__(self):
        return len(self.data_images)
    
class InitialMaskDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        Make pytorch Dataset for given task.

        Args:
            image_dir (str): dataset directory
            transform (torchvision.transforms) pytorch image transforms  
        """
        self.image_dir = image_dir
        
        # for positive patches 
        self.data_images = [x for x in os.listdir(image_dir) if int(x[:-4].split('_')[-1]) == 1]        
        self.transform = transform

    def __getitem__(self, idx):
        file_name = self.data_images[idx]
        img_path = os.path.join(self.image_dir, self.data_images[idx])
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, file_name
    
    def __len__(self):
        return len(self.data_images)

class SegmenterDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Make pytorch Dataset for given task.

        Args:
            image_dir (str): dataset directory
            mask_dir (str): dataset directory
            transform (albumentations transform): The albumentations transforms to be applied to images and masks. Defaults to None.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        # for positive patches 
        self.data_images = [x for x in os.listdir(image_dir) if int(x[:-4].split('_')[-1]) == 1]
        self.mask_images = os.listdir(mask_dir)    

    def __getitem__(self, idx):
        file_name = self.mask_images[idx]
        img_path = os.path.join(self.image_dir, self.mask_images[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_images[idx])  

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        mask[mask != 0] = 1 

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']
            
        return image, mask, file_name
    
    def __len__(self):
        return len(self.mask_images)