import torch
import numpy as np 
from tqdm import tqdm 
from typing import Tuple

from .metrics import Dice, IoU

def model_train(model,
                data_loader,
                criterion,
                optimizer,        
                device, 
                scheduler) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Train the model.

    Args:
        model (nn.Module): segmentation model 
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
    running_iouloss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    n_data = 0
    
    bceloss = criterion[0] 
    iouloss = criterion[1] 
    
    for X, y in tqdm(data_loader):
        optimizer.zero_grad()

        X_torch, y_torch = X.float().permute(0, 3, 1, 2).contiguous().to(device), y[..., 0].float().to(device)

        if type(model).__name__ == 'DeepLabV3':
            outputs = model(X_torch)['out']
        
        else:
            outputs = model(X_torch)
            
        outputs = torch.sigmoid(outputs)

        bce_loss = bceloss(outputs.squeeze(1), y_torch)
        iou_loss = iouloss(outputs.squeeze(1), y_torch)
        loss = bce_loss + iou_loss
        
        dice = Dice(outputs.squeeze(1), y_torch)
        iou = IoU(outputs.squeeze(1), y_torch)
        
        loss.backward()
        optimizer.step()
        
        ### update loss & metrics ###
        running_bceloss += bce_loss.item() * X.size(0)
        running_iouloss += iou_loss.item() * X.size(0)
        running_dice += dice.item() * X.size(0)
        running_iou += iou.item() * X.size(0)
        
        n_data += X.size(0)
    
    if scheduler:
        scheduler.step()
    
    ### Average loss & metrics ###
    avg_bce_loss = running_bceloss / n_data
    avg_iou_loss = running_iouloss / n_data
    avg_dice = running_dice / n_data
    avg_iou = running_iou / n_data 

    return avg_bce_loss, avg_iou_loss, avg_dice, avg_iou

def model_evaluate(model,
                   data_loader,
                   criterion,
                   device) -> Tuple[float, float, float, float]:
    """
    Evaluate the model.

    Args:
        model (nn.Module): DeepLabv3 model 
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
        running_dice = 0.0
        running_iou = 0.0
        
        bceloss = criterion[0] 
        iouloss = criterion[1] 
        
        for X, y in data_loader: 
            X_torch, y_torch = X.float().permute(0, 3, 1, 2).contiguous().to(device), y[..., 0].float().to(device)
            
            if type(model).__name__ == 'DeepLabV3':
                outputs = model(X_torch)['out']
            
            else:
                outputs = model(X_torch)
            
            outputs = torch.sigmoid(outputs)

            bce_loss = bceloss(outputs.squeeze(1), y_torch)
            iou_loss = iouloss(outputs.squeeze(1), y_torch)
            dice = Dice(outputs.squeeze(1), y_torch)
            iou = IoU(outputs.squeeze(1), y_torch)
                
            ### update loss & metrics ###
            running_bceloss += bce_loss.item() * X.size(0)
            running_iouloss += iou_loss.item() * X.size(0)
            running_dice += dice.item() * X.size(0)
            running_iou += iou.item() * X.size(0)
        
        ### Average loss & metrics ### 
        len_data = len(data_loader.dataset) 
        avg_bce_loss = running_bceloss / len_data
        avg_iou_loss = running_iouloss / len_data
        avg_dice = running_dice / len_data
        avg_iou = running_iou / len_data  

    return avg_bce_loss, avg_iou_loss, avg_dice, avg_iou

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
