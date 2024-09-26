import torch
from torchvision.transforms.functional import to_pil_image
import torchvision.models as models

import os
import numpy as np
from PIL import Image

import pandas as pd
from tqdm import tqdm

from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.utils.metrics import Dice, IoU
   
def make_mask(
    cls,
    model_type,
    morphology,
    data_loader,
    output_path,
    pmask_dir,
    device
) -> pd.DataFrame:
    """
    Make pseudo mask using CAM & SAM 

    Args:
        cls (nn.Module): classifier model 
        model_type (str): classifier model type
        morphology (bool): morphology operation
        data_loader (torch.DataLoader): pytorch dataloader
        output_path (str): save path 
        device (str): device 

    Returns:
        pd.DataFrame: result csv file 
    """
    save_dir = f'{output_path}'
    save_mask_dir = f'{output_path}/{pmask_dir}'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_mask_dir, exist_ok=True)
    
    cls.eval()
    
    output_dict = {
            'dice': [],
            'iou': []
        }
    
    for X, y, _, file_name in tqdm(data_loader):
        X_torch = X.float().permute(0, 3, 1, 2).contiguous().to(device)
        y_torch = y[..., 0].float().to(device)

        # Model input 
        batched_input = []
        
        for image, gt_mask, file in zip(X_torch, y_torch, file_name):
            if model_type == 'adl' or model_type == 'adl_fft':
                with torch.no_grad():
                    logit = cls(image.unsqueeze(0).div(255))
            else:
                logit = cls(image.unsqueeze(0).div(255))

            logit = torch.sigmoid(logit).squeeze()

            # for positive patches 
            if logit.item() >= 0:
                original_size = image.shape[1:3]
                
                cam_mask = cls.make_cam(to_pil_image(torch.as_tensor(image, dtype=torch.uint8).squeeze()), morphology=morphology, device=device)

                gt_mask = gt_mask.squeeze().cpu().numpy()

                dice = Dice(torch.from_numpy(gt_mask).unsqueeze(0), torch.from_numpy(cam_mask).unsqueeze(0))          
                iou = IoU(torch.from_numpy(gt_mask).unsqueeze(0), torch.from_numpy(cam_mask).unsqueeze(0))

                output_dict['iou'].append(iou)
                output_dict['dice'].append(dice)
                
                cam_mask = Image.fromarray(np.uint8(cam_mask))
                cam_mask.save(f'{save_mask_dir}/{file}')

    df = pd.DataFrame(output_dict)
    df.to_csv(f'{save_dir}/{pmask_dir}.csv')

    return df
