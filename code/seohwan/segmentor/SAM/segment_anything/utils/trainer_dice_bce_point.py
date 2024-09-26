import torch

import numpy as np 
from tqdm import tqdm 
from typing import Tuple
from torchvision.transforms.functional import to_pil_image

from .transforms import ResizeLongestSide
from .make_prompt import *
from .metrics import *

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAMPlusPlus

def model_train(model,
                data_loader,
                criterion,
                optimizer,        
                device,
                weight,
                scheduler) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Train the model.

    Args:
        model (nn.Module): SAM model 
        data_loader (torch.DataLoader): pytorch dataloader
        criterion (list): list of loss functions(BCE, IoU)
        optimizer (torch.optim.Optimzer): pytorch optimizer
        device (str): device
        scheduler (torch.optim.lr_scheduler): pytorch learning rate scheduler 

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: average losses(bce, iou), metrics(dice, iou)
    """
    
    # Training
    model.train()
    
    running_bceloss = 0.0
    # running_iouloss = 0.0
    running_diceloss = 0.0
    # running_focalloss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    n_data = 0
    
    # bceloss = criterion[0] 
    # iouloss = criterion[1]
    # diceloss = criterion[2]
    # focalloss = criterion[3]
    diceloss = criterion[0]
    bceloss = criterion[1]
    
    transform = ResizeLongestSide(target_length=model.image_encoder.img_size)
    
    # tqdm set 
    if (device == 0) or (device == 'cuda:1'):
        for X, y in tqdm(data_loader):
            optimizer.zero_grad()
            
            X_torch, y_torch = X.float().permute(0, 3, 1, 2).contiguous().to(device), y[..., 0].float().to(device)
            
            batched_input = []
            
            for image, mask in zip(X_torch, y_torch):
                # prepare image
                original_size = image.shape[1:3]
                
                image = transform.apply_image(image)
                image = torch.as_tensor(image, dtype=torch.float, device=device)
                image = image.permute(2, 0, 1).contiguous()
                
                point_coords = make_point_prompt(mask=mask.squeeze().cpu().numpy(), n_point=100)
                point_label = np.ones(shape=len(point_coords))
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=device)
                labels_torch = torch.as_tensor(point_label, dtype=torch.float, device=device)
                coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
            
                batched_input.append(
                    {
                        'image': image,
                        'point_coords': coords_torch,
                        'point_labels': labels_torch,
                        'original_size': original_size
                    }
                )
            
            batched_output = model(batched_input, multimask_output=False)
            
            loss = 0.0
            bce_loss = 0.0
            # iou_loss = 0.0
            dice_loss = 0.0
            # focal_loss = 0.0
            dice = 0.0
            iou = 0.0
            
            for i, gt_mask in enumerate(y_torch):
                
                masks = batched_output[i]['masks']
                masks_pred = batched_output[i]['masks_pred']
                
                ### loss & metrics ###
                bce_loss_ = bceloss(masks.squeeze(1), gt_mask.unsqueeze(0))
                # iou_loss_ = iouloss(masks.squeeze(1), gt_mask.unsqueeze(0))
                dice_loss_ = diceloss(masks.squeeze(1), gt_mask.unsqueeze(0)) 
                # focal_loss_ = focalloss(masks.squeeze(1), gt_mask.unsqueeze(0)) * weight
                # loss_ = dice_loss_ + focal_loss_
                loss_ = dice_loss_ + bce_loss_
                
                dice_ = Dice(masks_pred.squeeze(1), gt_mask.unsqueeze(0))
                iou_ = IoU(masks_pred.squeeze(1), gt_mask.unsqueeze(0))
                            
                loss = loss + loss_
                bce_loss = bce_loss + bce_loss_
                # iou_loss = iou_loss + iou_loss_
                dice_loss = dice_loss + dice_loss_
                # focal_loss = focal_loss + focal_loss_
                dice = dice + dice_
                iou = iou + iou_
            
            # average loss & metrcis (mini-batch)
            loss = loss / y_torch.shape[0]
            bce_loss = bce_loss / y_torch.shape[0]
            # iou_loss = iou_loss / y_torch.shape[0]
            dice_loss = dice_loss / y_torch.shape[0]
            # focal_loss = focal_loss / y_torch.shape[0]
            dice = dice / y_torch.shape[0]
            iou = iou / y_torch.shape[0]
                    
            loss.backward()
            optimizer.step()
                
            ### update loss & metrics ###
            running_bceloss += bce_loss * X.size(0)
            # running_iouloss += iou_loss * X.size(0)
            running_diceloss += dice_loss * X.size(0)
            # running_focalloss += focal_loss * X.size(0)
            running_dice += dice * X.size(0)
            running_iou += iou * X.size(0)
            
            n_data += X.size(0)
            
    else: 
        for X, y in tqdm(data_loader):
            optimizer.zero_grad()
            
            X_torch, y_torch = X.float().permute(0, 3, 1, 2).contiguous().to(device), y[..., 0].float().to(device)
            
            batched_input = []
            
            for image, mask in zip(X_torch, y_torch):
                # prepare image
                original_size = image.shape[1:3]
                
                image = transform.apply_image(image)
                image = torch.as_tensor(image, dtype=torch.float, device=device)
                image = image.permute(2, 0, 1).contiguous()
                
                point_coords = make_point_prompt(mask=mask.squeeze().cpu().numpy(), n_point=100)
                point_label = np.ones(shape=len(point_coords))
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=device)
                labels_torch = torch.as_tensor(point_label, dtype=torch.float, device=device)
                coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
            
                batched_input.append(
                    {
                        'image': image,
                        'point_coords': coords_torch,
                        'point_labels': labels_torch,
                        'original_size': original_size
                    }
                )
            
            batched_output = model(batched_input, multimask_output=False)
            
            loss = 0.0
            bce_loss = 0.0
            # iou_loss = 0.0
            dice_loss = 0.0
            # focal_loss = 0.0
            dice = 0.0
            iou = 0.0
            
            for i, gt_mask in enumerate(y_torch):
                
                masks = batched_output[i]['masks']
                masks_pred = batched_output[i]['masks_pred']
                
                ### loss & metrics ###
                bce_loss_ = bceloss(masks.squeeze(1), gt_mask.unsqueeze(0))
                # iou_loss_ = iouloss(masks.squeeze(1), gt_mask.unsqueeze(0)) 

                dice_loss_ = diceloss(masks.squeeze(1), gt_mask.unsqueeze(0)) 
                # focal_loss_ = focalloss(masks.squeeze(1), gt_mask.unsqueeze(0)) * weight
                # loss_ = dice_loss_ + focal_loss_
                loss_ = dice_loss_ + bce_loss_
                
                dice_ = Dice(masks_pred.squeeze(1), gt_mask.unsqueeze(0))
                iou_ = IoU(masks_pred.squeeze(1), gt_mask.unsqueeze(0))
                            
                loss = loss + loss_
                bce_loss = bce_loss + bce_loss_
                # iou_loss = iou_loss + iou_loss_
                dice_loss = dice_loss + dice_loss_
                # focal_loss = focal_loss + focal_loss_
                dice = dice + dice_
                iou = iou + iou_
            
            # average loss & metrcis (mini-batch)
            loss = loss / y_torch.shape[0]
            bce_loss = bce_loss / y_torch.shape[0]
            # iou_loss = iou_loss / y_torch.shape[0]
            dice_loss = dice_loss / y_torch.shape[0]
            # focal_loss = focal_loss / y_torch.shape[0]
            dice = dice / y_torch.shape[0]
            iou = iou / y_torch.shape[0]
                    
            loss.backward()
            optimizer.step()
                
            ### update loss & metrics ###
            running_bceloss += bce_loss * X.size(0)
            # running_iouloss += iou_loss * X.size(0)
            running_diceloss += dice_loss * X.size(0)
            # running_focalloss += focal_loss * X.size(0)
            running_dice += dice * X.size(0)
            running_iou += iou * X.size(0)
            
            n_data += X.size(0)
        
    if scheduler:
        scheduler.step()
    
    ### Average loss & metrics ###
    avg_bce_loss = running_bceloss / n_data
    # avg_iou_loss = running_iouloss / n_data
    avg_dice_loss = running_diceloss / n_data
    # avg_focal_loss = running_focalloss / n_data
    avg_dice = running_dice / n_data
    avg_iou = running_iou / n_data 

    # return avg_bce_loss, avg_iou_loss, avg_dice_loss, avg_focal_loss, avg_dice, avg_iou
    # return avg_dice_loss, avg_focal_loss, avg_dice, avg_iou
    return avg_dice_loss, avg_bce_loss, avg_dice, avg_iou

def model_evaluate(model,
                   cls,
                   data_loader,
                   criterion,
                   weight,
                   device) -> Tuple[float, float, float, float]:
    """
    Evaluate the model.

    Args:
        model (nn.Module): SAM model 
        data_loader (torch.DataLoader): pytorch dataloader
        criterion (list): list of loss functions(BCE, IoU)
        device (str): device 

    Returns:
        Tuple[float, float, float, float]: average losses(bce, iou), metrics(dice, iou)
    """
    
    # for p in cls.parameters():
    #     p.requires_grad = True

    # for p in cls.backbone[:-1].parameters():
    #     p.requires_grad = False

    # for p in cls.backbone[-1][:-1].parameters():
    #     p.requires_grad = False

    
    # Evaluation
    model.eval()
    cls.eval()
    
    # target_layers = [cls.backbone[-1][-1]]
    # targets = [ClassifierOutputTarget(0)]
    # gradcamplpl = GradCAMPlusPlus(model=cls, target_layers=target_layers)
    
    with torch.no_grad():
        
        running_bceloss = 0.0
        # running_iouloss = 0.0
        running_diceloss = 0.0
        # running_focalloss = 0.0
        running_dice = 0.0
        running_iou = 0.0
        
        # bceloss = criterion[0] 
        # iouloss = criterion[1]
        # diceloss = criterion[2]
        # focalloss = criterion[3]
        diceloss = criterion[0]
        bceloss = criterion[1]  
        
        transform = ResizeLongestSide(target_length=model.image_encoder.img_size)
        
        for X, y in tqdm(data_loader): 
            X_torch, y_torch = X.float().permute(0, 3, 1, 2).contiguous().to(device), y[..., 0].float().to(device)
            
            batched_input = []
            
            for image, mask in zip(X_torch, y_torch):
                # prepare image
                original_size = image.shape[1:3]
                
                # Bbox from CAM 
                cam_mask = cls.make_cam(to_pil_image(torch.as_tensor(image, dtype=torch.uint8).squeeze()), device=device)
                
                image = transform.apply_image(image)
                image = torch.as_tensor(image, dtype=torch.float, device=device)
                image = image.permute(2, 0, 1).contiguous()
                
                point_coords = make_point_prompt(mask=cam_mask.squeeze(), n_point=100)
                point_label = np.ones(shape=len(point_coords))
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=device)
                labels_torch = torch.as_tensor(point_label, dtype=torch.float, device=device)
                coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
            
                batched_input.append(
                    {
                        'image': image,
                        'point_coords': coords_torch,
                        'point_labels': labels_torch,
                        'original_size': original_size
                    }
                )

            batched_output = model(batched_input, multimask_output=False)
            
            bce_loss = 0.0
            # iou_loss = 0.0
            dice_loss = 0.0
            # focal_loss = 0.0
            dice = 0.0
            iou = 0.0
            
            for i, gt_mask in enumerate(y_torch):
                
                masks = batched_output[i]['masks']
                masks_pred = batched_output[i]['masks_pred']

                ### loss & metrics ###
                bce_loss_ = bceloss(masks.squeeze(1), gt_mask.unsqueeze(0))
                # iou_loss_ = iouloss(masks.squeeze(1), gt_mask.unsqueeze(0))
                dice_loss_ = diceloss(masks.squeeze(1), gt_mask.unsqueeze(0))
                # focal_loss_ = focalloss(masks.squeeze(1), gt_mask.unsqueeze(0)) * weight
                
                dice_ = Dice(masks_pred.squeeze(1), gt_mask.unsqueeze(0))
                iou_ = IoU(masks_pred.squeeze(1), gt_mask.unsqueeze(0))
                
                bce_loss = bce_loss + bce_loss_
                # iou_loss = iou_loss + iou_loss_
                dice_loss = dice_loss + dice_loss_
                # focal_loss = focal_loss + focal_loss_
                dice = dice + dice_
                iou = iou + iou_
                
            bce_loss = bce_loss / y_torch.shape[0]
            # iou_loss = iou_loss / y_torch.shape[0]
            dice_loss = dice_loss / y_torch.shape[0]
            # focal_loss = focal_loss / y_torch.shape[0]
            dice = dice / y_torch.shape[0]
            iou = iou / y_torch.shape[0]
                
            ### update loss & metrics ###
            running_bceloss += bce_loss.item() * X.size(0)
            # running_iouloss += iou_loss.item() * X.size(0)
            running_diceloss += dice_loss.item() * X.size(0)
            # running_focalloss += focal_loss.item() * X.size(0)
            running_dice += dice.item() * X.size(0)
            running_iou += iou.item() * X.size(0)
        
        ### Average loss & metrics ### 
        len_data = len(data_loader.dataset) 
        avg_bce_loss = running_bceloss / len_data
        # avg_iou_loss = running_iouloss / len_data
        avg_dice_loss = running_diceloss / len_data
        # avg_focal_loss = running_focalloss / len_data
        avg_dice = running_dice / len_data
        avg_iou = running_iou / len_data  

    # return avg_bce_loss, avg_iou_loss, avg_dice_loss, avg_focal_loss, avg_dice, avg_iou
    # return avg_dice_loss, avg_focal_loss, avg_dice, avg_iou
    return avg_dice_loss, avg_bce_loss, avg_dice, avg_iou


def model_evaluate_whole_bbox(model,
                   data_loader,
                   criterion,
                   device) -> Tuple[float, float, float, float]:
    """
    Evaluate the model.

    Args:
        model (nn.Module): SAM model 
        data_loader (torch.DataLoader): pytorch dataloader
        criterion (list): list of loss functions(BCE, IoU)
        device (str): device 

    Returns:
        Tuple[float, float, float, float]: average losses(bce, iou), metrics(dice, iou)
    """
    
    # Evaluation
    model.eval()
    
    with torch.no_grad():
        
        running_bceloss = 0.0
        running_iouloss = 0.0
        running_diceloss = 0.0
        running_focalloss = 0.0
        running_dice = 0.0
        running_iou = 0.0
        
        bceloss = criterion[0] 
        iouloss = criterion[1]
        diceloss = criterion[2]
        focalloss = criterion[3] 
        
        transform = ResizeLongestSide(target_length=model.image_encoder.img_size)
        
        for X, y in tqdm(data_loader): 
            X_torch, y_torch = X.float().permute(0, 3, 1, 2).contiguous().to(device), y[..., 0].float().to(device)
            
            batched_input = []
            
            for image, mask in zip(X_torch, y_torch):
                # prepare image
                original_size = image.shape[1:3]
                
                image = transform.apply_image(image)
                image = torch.as_tensor(image, dtype=torch.float, device=device)
                image = image.permute(2, 0, 1).contiguous()
                
                box = make_whole_box_prompt(
                    mask=mask.cpu().numpy(), 
                    scale_factor=1.0,
                    return_xyxy=True
                )

                box = transform.apply_boxes(box, original_size)
                box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
                box_torch = box_torch[None, :]
            
                batched_input.append(
                    {
                        'image': image,
                        'boxes': box_torch,
                        'original_size': original_size
                    }
                )

            batched_output = model(batched_input, multimask_output=False)

            bce_loss = 0.0
            iou_loss = 0.0
            dice_loss = 0.0
            focal_loss = 0.0
            dice = 0.0
            iou = 0.0
            
            for i, gt_mask in enumerate(y_torch):
                
                masks = batched_output[i]['masks']
                masks_pred = batched_output[i]['masks_pred']
                
                ### loss & metrics ###
                bce_loss_ = bceloss(masks.squeeze(1), gt_mask.unsqueeze(0))
                iou_loss_ = iouloss(masks.squeeze(1), gt_mask.unsqueeze(0))
                dice_loss_ = diceloss(masks.squeeze(1), gt_mask.unsqueeze(0)) 
                focal_loss_ = focalloss(masks.squeeze(1), gt_mask.unsqueeze(0)) * weight
                
                dice_ = Dice(masks_pred.squeeze(1), gt_mask.unsqueeze(0))
                iou_ = IoU(masks_pred.squeeze(1), gt_mask.unsqueeze(0))
                
                bce_loss = bce_loss + bce_loss_
                iou_loss = iou_loss + iou_loss_
                dice_loss = dice_loss + dice_loss_
                focal_loss = focal_loss + focal_loss_
                dice = dice + dice_
                iou = iou + iou_
                
            bce_loss = bce_loss / y_torch.shape[0]
            iou_loss = iou_loss / y_torch.shape[0]
            dice_loss = dice_loss / y_torch.shape[0]
            focal_loss = focal_loss / y_torch.shape[0]
            dice = dice / y_torch.shape[0]
            iou = iou / y_torch.shape[0]
                
            ### update loss & metrics ###
            running_bceloss += bce_loss.item() * X.size(0)
            running_iouloss += iou_loss.item() * X.size(0)
            running_diceloss += dice_loss.item() * X.size(0)
            running_focalloss += focal_loss.item() * X.size(0)
            running_dice += dice.item() * X.size(0)
            running_iou += iou.item() * X.size(0)
        
        ### Average loss & metrics ### 
        len_data = len(data_loader.dataset) 
        avg_bce_loss = running_bceloss / len_data
        avg_iou_loss = running_iouloss / len_data
        avg_dice_loss = running_diceloss / len_data
        avg_focal_loss = running_focalloss / len_data
        avg_dice = running_dice / len_data
        avg_iou = running_iou / len_data  

    return avg_bce_loss, avg_iou_loss, avg_dice_loss, avg_focal_loss, avg_dice, avg_iou





class EarlyStopping:
    def __init__(self, patience=10, delta=0.0, mode='min', verbose=True):
        """
        Pytorch Early Stopping

        Args:
            patience (int, optional): patience. Defaults to 10.
            delta (float, optional): threshold to update best score. Defaults to 0.0.
            mode (str, optional): 'min' or 'max'. Defaults to 'min'(comparing loss -> lower is better).
            verbose (bool, optional): verbose. Defaults to True.
        """
        self.early_stop = False
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        
        self.best_score = np.Inf if mode == 'min' else 0
        self.mode = mode
        self.delta = delta
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            self.counter = 0
            
        elif self.mode == 'min':
            if score < (self.best_score - self.delta):
                self.counter = 0
                self.best_score = score
                if self.verbose:
                    print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f}')
            else:
                self.counter += 1
                if self.verbose:
                    print(f'[EarlyStopping] (Patience) {self.counter}/{self.patience}, ' \
                          f'Best: {self.best_score:.5f}' \
                          f', Current: {score:.5f}, Delta: {np.abs(self.best_score - score):.5f}')
                
        elif self.mode == 'max':
            if score > (self.best_score + self.delta):
                self.counter = 0
                self.best_score = score
                if self.verbose:
                    print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f}')
            else:
                self.counter += 1
                if self.verbose:
                    print(f'[EarlyStopping] (Patience) {self.counter}/{self.patience}, ' \
                          f'Best: {self.best_score:.5f}' \
                          f', Current: {score:.5f}, Delta: {np.abs(self.best_score - score):.5f}')
                
        if self.counter >= self.patience:
            if self.verbose:
                print(f'[EarlyStop Triggered] Best Score: {self.best_score:.5f}')
            # Early Stop
            self.early_stop = True
        else:
            # Continue
            self.early_stop = False
