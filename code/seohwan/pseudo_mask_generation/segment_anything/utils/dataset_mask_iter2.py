import os
import numpy as np 
from PIL import Image
from typing import Type

import torch
from torch.utils.data import Dataset
import torchvision.transforms as tr 
import cv2

class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, percen80_dir, transform):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        # self.data_images = [x for x in os.listdir(image_dir) if int(x[:-4].split('_')[-1]) == 1]
        self.mask_images = os.listdir(mask_dir)
        self.pseudo_80percen = os.listdir(percen80_dir)
        self.labels = []
        
        for file in self.pseudo_80percen:
            if file.endswith('.png'):
                label = int(file[:-4].split('_')[-1])
                self.labels.append(label)
                
        self.transform = transform

    def __getitem__(self, idx):
        file_name = self.pseudo_80percen[idx]
        img_path = os.path.join(self.image_dir, file_name)
        mask_path = os.path.join(self.mask_dir, file_name)  
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = Image.fromarray(mask)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

            
        label = torch.tensor(label)
        
        return np.array(image), np.array(mask), label, file_name
    
    def __len__(self):
        return len(self.pseudo_80percen)
    
def make_dataset(
    image_dir: str,
    mask_dir: str,
    percen80_dir: str,
    transform = tr.Resize(512)
) -> Type[torch.utils.data.Dataset]:
    """
    Make pytorch Dataset for given task.
    Read the image using the PIL library and return it as an np.array.

    Args:
        image_dir (str): dataset directory
        transform (torchvision.transforms) pytorch image transforms  

    Returns:
        torch.Dataset: pytorch Dataset
    """
        
    dataset = CustomDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        percen80_dir=percen80_dir,
        transform=transform
    )
            
    return dataset