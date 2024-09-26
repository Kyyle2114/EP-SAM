import torch
from torchvision.transforms.functional import to_pil_image

import pandas as pd 
import numpy as np
from tqdm import tqdm 
import os 
from PIL import Image 

from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.utils.metrics import Dice, IoU
from segment_anything.utils.box_metrics import calculate_dice, calculate_iou
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAMPlusPlus, LayerCAM, GradCAM, ScoreCAM

def make_box(
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
    
    if model_type == 'gradcamplpl':
        for p in cls.parameters():
            p.requires_grad = True

        for p in cls.backbone[:-1].parameters():
            p.requires_grad = False

        for p in cls.backbone[-1][:-1].parameters():
            p.requires_grad = False
        
        cls.eval()
        
        target_layers = [cls.backbone[-1][-1]]
        targets = [ClassifierOutputTarget(0)]
        gradcamplpl = GradCAMPlusPlus(model=cls, target_layers=target_layers)

    elif model_type == 'layercam':    
        for p in cls.parameters():
            p.requires_grad = True

        for p in cls.backbone[:-1].parameters():
            p.requires_grad = False

        for p in cls.backbone[-1][:-1].parameters():
            p.requires_grad = False   
             
        cls.eval()
        
        target_layers_for_layercam = [cls.backbone[-4][-1], cls.backbone[-3][-1], cls.backbone[-2][-1], cls.backbone[-1][-1]]
        targets = [ClassifierOutputTarget(0)]
        layercam = LayerCAM(model=cls, target_layers=target_layers_for_layercam) 

    else:
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
                
                # Bbox from CAM
                if model_type == 'adl' or model_type == 'adl_fft': 
                    cam_mask = cls.make_cam(to_pil_image(torch.as_tensor(image, dtype=torch.uint8).squeeze()), morphology=morphology, device=device)
                elif model_type == 'gradcamplpl':                    
                    cam_mask = cls.make_cam(to_pil_image(torch.as_tensor(image, dtype=torch.uint8).squeeze()), morphology=morphology, device=device, cam=gradcamplpl, targets=targets, cls=cls)
                elif model_type == 'layercam':
                    cam_mask = cls.make_cam(to_pil_image(torch.as_tensor(image, dtype=torch.uint8).squeeze()), morphology=morphology, device=device, cam=layercam, targets=targets, cls=cls)

                cam_bbox = cls.make_bbox(cam_mask)
                gt_bbox = cls.make_bbox(gt_mask.cpu().numpy())
                
                bbox_iou = calculate_iou(cam_bbox, gt_bbox)
                bbox_dice = calculate_dice(cam_bbox, gt_bbox)
                
            
                output_dict['iou'].append(bbox_iou)
                output_dict['dice'].append(bbox_dice)

    df = pd.DataFrame(output_dict)
    df.to_csv(f'{save_dir}/{pmask_dir}.csv')

    return df