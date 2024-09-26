import os
from PIL import Image
from typing import Type

import torch
from torch.utils.data import Dataset

class CustomClassifierDataset(Dataset):
    def __init__(self, image_dir, transform):
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
    
class CustomSegmenterDataset(Dataset):
    def __init__(self, image_dir, transform):
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
    
def make_classifier_dataset(
    image_dir: str,
    transform=None
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
        
    dataset = CustomClassifierDataset(
        image_dir=image_dir,
        transform=transform
    )
            
    return dataset
    
def make_segmenter_dataset(
    image_dir: str,
    transform=None
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
        
    dataset = CustomSegmenterDataset(
        image_dir=image_dir,
        transform=transform
    )
            
    return dataset