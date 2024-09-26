import os
from typing import Type
import cv2
from tqdm import tqdm 
import torch
from torch.utils.data import Dataset, DataLoader

from segment_anything.utils.metrics import Dice, IoU 

class CustomDataset(Dataset):
    def __init__(self, mask_dir, pmask_dir):
        self.mask_dir = mask_dir
        self.pmask_dir = pmask_dir
        self.pmasks = os.listdir(pmask_dir)
        
    def __getitem__(self, idx):
        
        mask_path = os.path.join(self.mask_dir, self.pmasks[idx])
        pmask_path = os.path.join(self.pmask_dir, self.pmasks[idx])  

        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        pmask = cv2.imread(pmask_path)
        pmask = cv2.cvtColor(pmask, cv2.COLOR_BGR2RGB)
        
        return mask[..., 0], pmask[..., 0]
    
    def __len__(self):
        return len(self.pmasks)
    
def make_dataset(
    mask_dir: str,
    pmask_dir: str
) -> Type[torch.utils.data.Dataset]:

    dataset = CustomDataset(
        mask_dir=mask_dir,
        pmask_dir=pmask_dir,
    )
            
    return dataset

def compare_mask(
    mask_dir: str,
    pmask_dir: str
):
    """
    Calculate the IoU and Dice between the GT mask and Pseudo mask.

    Args:
        mask_dir (str): gt mask directory
        pmask_dir (str): pseudo mask directory
    """
    mask_set = make_dataset(
        mask_dir=mask_dir,
        pmask_dir=pmask_dir
    )

    data_loader = DataLoader(
        mask_set, 
        batch_size=1,
        shuffle=False
    )

    dice = 0.0
    iou = 0.0 
    cnt = 0 

    for mask, pmask in tqdm(data_loader):
        
        dice_ = Dice(pmask, mask)
        iou_ = IoU(pmask, mask)

        dice += dice_.item()
        iou += iou_.item()
        
        cnt += 1
        
    print(f'Mask : {cnt} files')
    print(f'Dice : {dice / cnt}')
    print(f'IoU : {iou / cnt}')
            