import torch
from torchvision.transforms.functional import to_pil_image

import pandas as pd 
import numpy as np
from tqdm import tqdm 
import os 
from PIL import Image 

from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.utils.metrics import Dice, IoU

def make_pseudo_mask(
    # sam,
    cls,
    data_loader,
    output_path,
    device
) -> pd.DataFrame:
    """
    Make pseudo mask using CAM & SAM 

    Args:
        sam (nn.Module): SAM model 
        cls (nn.Module): classifier model 
        data_loader (torch.DataLoader): pytorch dataloader
        output_path (str): save path 
        device (str): device 

    Returns:
        pd.DataFrame: result csv file 
    """
    save_dir = f'{output_path}'
    save_mask_dir = f'{output_path}/test_pseudo_masks'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_mask_dir, exist_ok=True)
    
    # sam.eval()
    cls.eval()
    
    with torch.no_grad():
        
        # transform = ResizeLongestSide(target_length=sam.image_encoder.img_size)
        
        output_dict = {
                'file': [],
                'os_score': [],
                'op_score': [],
                'f1': [],
                'dice': [],
                'iou': []
            }
        
        for X, _, file_name in tqdm(data_loader):
            X_torch = X.float().permute(0, 3, 1, 2).contiguous().to(device)
            
            # Model input 
            batched_input = []
            
            for image, file in zip(X_torch, file_name):
                
                logit = cls(image.unsqueeze(0).div(255))
                logit = torch.sigmoid(logit).squeeze()
                
                # for positive patches 
                if logit.item() > 0.5:
                    original_size = image.shape[1:3]
                    
                    # Bbox from CAM 
                    cam_mask = cls.make_cam(to_pil_image(torch.as_tensor(image, dtype=torch.uint8).squeeze()), device=device)
                    mask_to_save = Image.fromarray(cam_mask.astype(np.uint8))
                    
                else:
                    # Classifier predicts negative, create a full-size mask of zeros
                    full_mask = np.zeros((image.shape[1], image.shape[2]), dtype=np.uint8)  # Full mask of zeros
                    mask_to_save = Image.fromarray(full_mask)
                
                mask_to_save.save(os.path.join(save_mask_dir, f'{file}'))
                

