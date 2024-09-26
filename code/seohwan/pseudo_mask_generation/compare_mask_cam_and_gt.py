import torch
from torchvision.transforms.functional import to_pil_image
import torchvision.models as models
from model.backbone import resnet50

import os
import numpy as np
from PIL import Image
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
import pandas as pd
from tqdm import tqdm
from adl.fft import extract_freq_components

from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.utils.metrics import Dice, IoU
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAMPlusPlus, LayerCAM, GradCAM, EigenCAM

def apply_dense_crf(image, cam, params):
    cam_normalized = (cam * 255).astype(np.uint8)
    cam_normalized[cam_normalized > 0] = 1
    
    d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], 2)
    U = unary_from_labels(cam_normalized, 2, gt_prob=params['gt_prob'], zero_unsure=False)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=params['gaussian_sxy'], compat=params['gaussian_compat'])
    d.addPairwiseBilateral(sxy=params['bilateral_sxy'], srgb=params['bilateral_srgb'], 
                           rgbim=image, compat=params['bilateral_compat'])
    Q = d.inference(params['inference_steps'])
    
    result = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1]))
    return result
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
    
    if model_type == 'gradcam':
        for p in cls.parameters():
            p.requires_grad = True

        for p in cls.backbone[:-1].parameters():
            p.requires_grad = False

        for p in cls.backbone[-1][:-1].parameters():
            p.requires_grad = False
        
        cls.eval()
        
        target_layers = [cls.backbone[-1][-1]]
        targets = [ClassifierOutputTarget(0)]
        gradcam = GradCAMPlusPlus(model=cls, target_layers=target_layers)

    elif model_type == 'layercam':    
        for p in cls.parameters():
            p.requires_grad = True

        cls.eval()
        
        target_layers_for_layercam = [cls.backbone[-3][-1], cls.backbone[-1][-1]]
        targets = [ClassifierOutputTarget(0)]
        layercam = LayerCAM(model=cls, target_layers=target_layers_for_layercam) 

    elif model_type == 'eigencam':
        for p in cls.parameters():
            p.requires_grad = True

        for p in cls.backbone[:-1].parameters():
            p.requires_grad = False

        for p in cls.backbone[-1][:-1].parameters():
            p.requires_grad = False
            
        cls.eval()
        
        target_layers = [cls.backbone[-1][-1]]
        targets = [ClassifierOutputTarget(0)]
        eigencam = EigenCAM(model=cls, target_layers=target_layers)
    
    
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
                elif model_type == 'gradcam':                    
                    cam_mask = cls.make_cam(to_pil_image(torch.as_tensor(image, dtype=torch.uint8).squeeze()), morphology=morphology, device=device, cam=gradcam, targets=targets, cls=cls)
                elif model_type == 'layercam':
                    cam_mask = cls.make_cam(to_pil_image(torch.as_tensor(image, dtype=torch.uint8).squeeze()), morphology=morphology, device=device, cam=layercam, targets=targets, cls=cls)
                elif model_type == 'eigencam':
                    cam_mask = cls.make_cam(to_pil_image(torch.as_tensor(image, dtype=torch.uint8).squeeze()), morphology=morphology, device=device, cam=eigencam, targets=targets, cls=cls)

                # Apply Dense CRF
                params = {
                    'gaussian_sxy': 3,
                    'gaussian_compat': 1,
                    'bilateral_sxy': 30,
                    'bilateral_srgb': 3,
                    'bilateral_compat': 3,
                    'gt_prob': 0.7,
                    'inference_steps': 5
                }
                crf_mask = apply_dense_crf(np.array(to_pil_image(torch.as_tensor(image, dtype=torch.uint8).squeeze())), cam_mask, params)

                gt_mask = gt_mask.squeeze().cpu().numpy()

                dice = Dice(torch.from_numpy(gt_mask).unsqueeze(0), torch.from_numpy(crf_mask).unsqueeze(0))          
                iou = IoU(torch.from_numpy(gt_mask).unsqueeze(0), torch.from_numpy(crf_mask).unsqueeze(0))

                output_dict['iou'].append(iou)
                output_dict['dice'].append(dice)

                cam_mask = Image.fromarray(np.uint8(crf_mask))
                cam_mask.save(f'{save_mask_dir}/{file}')

    df = pd.DataFrame(output_dict)
    df.to_csv(f'{save_dir}/{pmask_dir}.csv')

    return df    
# def make_mask(
#     cls,
#     model_type,
#     morphology,
#     data_loader,
#     output_path,
#     pmask_dir,
#     device
# ) -> pd.DataFrame:
#     """
#     Make pseudo mask using CAM & SAM 

#     Args:
#         cls (nn.Module): classifier model 
#         model_type (str): classifier model type
#         morphology (bool): morphology operation
#         data_loader (torch.DataLoader): pytorch dataloader
#         output_path (str): save path 
#         device (str): device 

#     Returns:
#         pd.DataFrame: result csv file 
#     """
#     save_dir = f'{output_path}'
#     save_mask_dir = f'{output_path}/{pmask_dir}'
#     os.makedirs(save_dir, exist_ok=True)
#     os.makedirs(save_mask_dir, exist_ok=True)
    
#     if model_type == 'gradcam':
#         for p in cls.parameters():
#             p.requires_grad = True

#         for p in cls.backbone[:-1].parameters():
#             p.requires_grad = False

#         for p in cls.backbone[-1][:-1].parameters():
#             p.requires_grad = False
        
#         cls.eval()
        
#         target_layers = [cls.backbone[-1][-1]]
#         targets = [ClassifierOutputTarget(0)]
#         gradcam = GradCAMPlusPlus(model=cls, target_layers=target_layers)

#     elif model_type == 'layercam':    
#         for p in cls.parameters():
#             p.requires_grad = True

#         # for p in cls.backbone[:-1].parameters():
#         #     p.requires_grad = False

#         # for p in cls.backbone[-1][:-1].parameters():
#         #     p.requires_grad = False   
             
#         cls.eval()
        
#         target_layers_for_layercam = [cls.backbone[-3][-1], cls.backbone[-1][-1]]
#         targets = [ClassifierOutputTarget(0)]
#         layercam = LayerCAM(model=cls, target_layers=target_layers_for_layercam) 

#     elif model_type == 'eigencam':
#         for p in cls.parameters():
#             p.requires_grad = True

#         for p in cls.backbone[:-1].parameters():
#             p.requires_grad = False

#         for p in cls.backbone[-1][:-1].parameters():
#             p.requires_grad = False
            
#         cls.eval()
        
#         target_layers = [cls.backbone[-1][-1]]
#         targets = [ClassifierOutputTarget(0)]
#         eigencam = EigenCAM(model=cls, target_layers=target_layers)
    
    
#     else:
#         cls.eval()
    
#     output_dict = {
#             'dice': [],
#             'iou': []
#         }
    
#     for X, y, _, file_name in tqdm(data_loader):
#         X_torch = X.float().permute(0, 3, 1, 2).contiguous().to(device)
#         y_torch = y[..., 0].float().to(device)

#         # Model input 
#         batched_input = []
        
#         for image, gt_mask, file in zip(X_torch, y_torch, file_name):
#             if model_type == 'adl' or model_type == 'adl_fft':
#                 with torch.no_grad():
#                     logit = cls(image.unsqueeze(0).div(255))
#             else:
#                 logit = cls(image.unsqueeze(0).div(255))

#             logit = torch.sigmoid(logit).squeeze()

#             # for positive patches 
#             if logit.item() >= 0:
#                 original_size = image.shape[1:3]

#                 # Bbox from CAM
#                 if model_type == 'adl' or model_type == 'adl_fft': 
#                     cam_mask = cls.make_cam(to_pil_image(torch.as_tensor(image, dtype=torch.uint8).squeeze()), morphology=morphology, device=device)
#                 elif model_type == 'gradcam':                    
#                     cam_mask = cls.make_cam(to_pil_image(torch.as_tensor(image, dtype=torch.uint8).squeeze()), morphology=morphology, device=device, cam=gradcam, targets=targets, cls=cls)
#                 elif model_type == 'layercam':
#                     cam_mask = cls.make_cam(to_pil_image(torch.as_tensor(image, dtype=torch.uint8).squeeze()), morphology=morphology, device=device, cam=layercam, targets=targets, cls=cls)
#                 elif model_type == 'eigencam':
#                     cam_mask = cls.make_cam(to_pil_image(torch.as_tensor(image, dtype=torch.uint8).squeeze()), morphology=morphology, device=device, cam=eigencam, targets=targets, cls=cls)

#                 gt_mask = gt_mask.squeeze().cpu().numpy()

#                 dice = Dice(torch.from_numpy(gt_mask).unsqueeze(0), torch.from_numpy(cam_mask).unsqueeze(0))          
#                 iou = IoU(torch.from_numpy(gt_mask).unsqueeze(0), torch.from_numpy(cam_mask).unsqueeze(0))

#                 output_dict['iou'].append(iou)
#                 output_dict['dice'].append(dice)
                
#                 cam_mask = Image.fromarray(np.uint8(cam_mask))
#                 cam_mask.save(f'{save_mask_dir}/{file}')

#     df = pd.DataFrame(output_dict)
#     df.to_csv(f'{save_dir}/{pmask_dir}.csv')

#     return df
