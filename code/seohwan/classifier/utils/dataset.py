import os
from PIL import Image
from typing import Type

import torch
import torchvision.transforms as tr
from torch.utils.data import Dataset

class CustomDataset(Dataset):
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
    
def make_dataset(image_dir: str,
                 transform=tr.Compose([tr.Resize(512), 
                                       tr.RandomHorizontalFlip(), 
                                       tr.RandomVerticalFlip(), 
                                       tr.RandomRotation(10), 
                                       tr.ToTensor()])
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
        
    dataset = CustomDataset(image_dir=image_dir,
                            transform=transform)
            
    return dataset