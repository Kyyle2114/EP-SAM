import torch
from torchvision.transforms.functional import to_pil_image

import pandas as pd 
import numpy as np
from tqdm import tqdm 
import os 
from PIL import Image 
from segment_anything.utils.fft import extract_freq_components
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.utils.metrics import Dice, IoU

def make_pseudo_mask(
    sam,
    cls,
    data_loader,
    output_path,
    pmask_dir,
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
    save_mask_dir = f'{output_path}/{pmask_dir}'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_mask_dir, exist_ok=True)
    
    sam.eval()
    cls.eval()
    
    with torch.no_grad():
        
        transform = ResizeLongestSide(target_length=sam.image_encoder.img_size)
        
        output_dict = {
                'file': [],
                'os_score': [],
                'op_score': [],
                'f1': [],
                'dice': [],
                'iou': []
            }
        
        for X, y, _, file_name in tqdm(data_loader):
            X_torch = X.float().permute(0, 3, 1, 2).contiguous().to(device)
            
            y_torch = y[..., 0].float().to(device)
            
            # Model input 
            batched_input = []
            
            for image, gt_mask, file in zip(X_torch, y_torch, file_name):
                
                logit = cls(image.unsqueeze(0).div(255))
                logit = torch.sigmoid(logit).squeeze()
                
                # for positive patches 
                if logit.item() >= 0:
                    original_size = image.shape[1:3]
                    
                    # Bbox from CAM 
                    cam_mask = cls.make_cam(to_pil_image(torch.as_tensor(image, dtype=torch.uint8).squeeze()), device=device)
                    cam_bbox = cls.make_bbox(cam_mask)
                    
                    box = transform.apply_boxes(cam_bbox, original_size)
                    box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
                    box_torch = box_torch[None, :]
                    
                    # SAM input 
                    image = transform.apply_image(image)
                    image = torch.as_tensor(image, dtype=torch.float, device=device)
                    image = image.permute(2, 0, 1).contiguous()
                
                    batched_input.append(
                        {
                            'image': image,
                            'boxes': box_torch,
                            'original_size': original_size,
                            'cam_mask': cam_mask,
                            'gt_mask': gt_mask,
                            'file_name': file
                        }
                    )
                    
            
            if len(batched_input) > 0:
                batched_output = sam(batched_input, multimask_output=False)
                
            else:
                continue
            
            for output in batched_output:
                mask_pred = output['masks_pred'].squeeze().cpu().numpy()
                gt_mask = output['gt_mask'].squeeze().cpu().numpy()
                mask_cam = output['cam_mask']
                file = output['file_name']

                os_score = (mask_cam & mask_pred).sum() / (mask_pred.sum() + 1)
                op_score = (mask_cam & mask_pred).sum() / (mask_cam.sum() + 1)
                f1 = (2 * os_score * op_score) / (os_score + op_score)
                dice = Dice(torch.from_numpy(gt_mask).unsqueeze(0), torch.from_numpy(mask_pred).unsqueeze(0))          
                iou = IoU(torch.from_numpy(gt_mask).unsqueeze(0), torch.from_numpy(mask_pred).unsqueeze(0))
                
                output_dict['file'].append(file)
                output_dict['os_score'].append(os_score)
                output_dict['op_score'].append(op_score)
                output_dict['f1'].append(f1)
                output_dict['dice'].append(dice.item())
                output_dict['iou'].append(iou.item())
                
                # save pseudo mask 
                pseudo_mask = Image.fromarray(np.uint8(mask_pred))
                pseudo_mask.save(f'{save_mask_dir}/{file}')
                cam_mask = Image.fromarray(np.uint8(mask_cam))
                cam_mask.save(f'{save_dir}/internel_train_cam_masks_test/{file}')
                
        df = pd.DataFrame(output_dict)
        df.to_csv(f'{save_dir}/{pmask_dir}.csv')

    return df