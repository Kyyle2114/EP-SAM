import os
import numpy as np 
import pandas as pd
from PIL import Image
from typing import Type

import torch
from torch.utils.data import Dataset
import torchvision.transforms as tr 

class CustomDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.image_dir = image_dir
        self.data_images = [x for x in os.listdir(image_dir) if int(x[:-4].split('_')[-1]) == 1]
        self.labels = []
        
        for file in self.data_images:
            if file.endswith('.png'):
                label = int(file[:-4].split('_')[-1])
                self.labels.append(label)
                
        self.transform = transform

    def __getitem__(self, idx):
        file_name = self.data_images[idx]
        img_path = os.path.join(self.image_dir, file_name)
        
        image = Image.open(img_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(label)
        
        return np.array(image), label, file_name
    
    def __len__(self):
        return len(self.data_images)
    
def make_dataset(
    image_dir: str,
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
        transform=transform
    )
            
    return dataset