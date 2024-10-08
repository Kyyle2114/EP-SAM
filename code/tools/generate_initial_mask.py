import torch
from torchvision.transforms.functional import to_pil_image

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
   
def generate_initial_mask(
    classifier,
    data_loader,
    output_path,
    device,
    dataset_type
):
    """
    Make pseudo mask using enhanced ADL CAM

    Args:
        classifier (nn.Module): classifier model 
        data_loader (torch.DataLoader): pytorch dataloader
        output_path (str): save path 
        device (str): device 
        dataset_type (str): dataset type (camelyon16 or camelyon17)
    """
    save_dir = f'{output_path}/initial_mask'
    os.makedirs(save_dir, exist_ok=True)
    
    classifier.eval()
    
    for X, file_name in tqdm(data_loader):
        X_torch = X.float().permute(0, 3, 1, 2).contiguous().to(device)
        
        for image, file in zip(X_torch, file_name):
                
            cam_mask, _ = classifier.generate_cam_masks(
                image=to_pil_image(torch.as_tensor(image, dtype=torch.uint8).squeeze()),
                device=device,
                dataset_type=dataset_type
            )
            
            cam_mask = Image.fromarray(np.uint8(cam_mask))
            cam_mask.save(f'{save_dir}/{file}')

    return 
