import torch
from torchvision.transforms.functional import to_pil_image
 
import numpy as np
from tqdm import tqdm 
import os 
from PIL import Image 

from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.utils.make_prompt import *
   
def generate_sam_mask(
    sam,
    classifier,
    data_loader,
    output_path,
    iter,
    device,
    dataset_type
):
    """
    Make pseudo mask using enhance ADL CAM & SAM 

    Args:
        sam (nn.Module) SAM model 
        classifier (nn.Module): classifier model 
        data_loader (torch.DataLoader): pytorch dataloader
        output_path (str): save path 
        iter (int): number of iteration
        device (str): device 
        dataset_type (str): dataset type (camelyon16 or camelyon17)
    """
    save_dir = f'{output_path}/iter_{iter}'
    os.makedirs(save_dir, exist_ok=True)
    
    sam.eval()
    classifier.eval()
    
    transform = ResizeLongestSide(target_length=sam.image_encoder.img_size)

    for X, y, file_name in tqdm(data_loader):
        X_torch = X.float().permute(0, 3, 1, 2).contiguous().to(device)
        y_torch = y[..., 0].float().to(device)

        batched_input = []
        
        for image, gt_mask, file in zip(X_torch, y_torch, file_name):
        
            original_size = image.shape[1:3]
            
            cam_mask, normalized_cam = classifier.generate_cam_masks(
                image=to_pil_image(torch.as_tensor(image, dtype=torch.uint8).squeeze()), 
                device=device,
                dataset_type=dataset_type
            )

            # SAM input 
            image = transform.apply_image(image)
            image = torch.as_tensor(image, dtype=torch.float, device=device)
            image = image.permute(2, 0, 1).contiguous()

            point_coords = make_proba_point_prompt(softmax_cam=normalized_cam, cam_mask=cam_mask, n_point=50)
            point_label = np.ones(shape=len(point_coords))
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=device)
            labels_torch = torch.as_tensor(point_label, dtype=torch.float, device=device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]

            batched_input.append(
                {
                    'image': image,
                    'original_size': original_size,
                    'point_coords': coords_torch,
                    'point_labels': labels_torch,
                    'cam_mask': cam_mask,
                    'gt_mask': gt_mask,
                    'file_name': file                        
                }
            )
        
        batched_output = sam(batched_input, multimask_output=False)    
        
        for output in batched_output:
            mask_pred = output['masks_pred'].squeeze().cpu().numpy()
            gt_mask = output['gt_mask'].squeeze().cpu().numpy()
            mask_cam = output['cam_mask']
            file = output['file_name']
            
            # ids score 
            threshold = 0.9
            ids_score = (mask_cam & mask_pred).sum() / (mask_pred.sum() + 1)
            
            if ids_score >= threshold:
                # save pseudo mask 
                pseudo_mask = Image.fromarray(np.uint8(mask_pred))
                pseudo_mask.save(f'{save_dir}/{file}')
            
    return 