import torch
import matplotlib.pyplot as plt 
import numpy as np 

from tqdm import tqdm 
from typing import Tuple

def model_train(model,
                data_loader,
                criterion,
                optimizer,        
                device, 
                scheduler) -> Tuple[float, float]:
    """
    Train the model for binary classification. 

    Args:
        model (nn.Module): pytorch model 
        data_loader (torch.DataLoader): pytorch dataloader
        criterion (nn.Module): pytorch loss 
        optimizer (torch.optim.Optimzer): pytorch optimizer
        device (str): device
        scheduler (torch.optim.lr_scheduler): pytorch learning rate scheduler 

    Returns:
        Tuple[float, float]: average loss & accuracy
    """
    
    model.train()
    
    running_loss = 0
    correct = 0

    for X, y in tqdm(data_loader):
        X, y = X.to(device), y.float().to(device)
        
        optimizer.zero_grad()
        
        output = model(X)
        output = output.squeeze()
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        # binary classification
        output = torch.sigmoid(output)
        pred = output >= torch.FloatTensor([0.5]).to(device)
        correct += pred.eq(y).sum().item()  
        running_loss += loss.item() * X.size(0)
    
    if scheduler:
        scheduler.step() 
        
    accuracy = correct / len(data_loader.dataset) # Avg acc
    loss = running_loss / len(data_loader.dataset) # Avg loss
 
    return loss, accuracy


def model_evaluate(model,
                   data_loader,
                   criterion,
                   device) -> Tuple[float, float]:
    """
    Evaluate the model for binary classification,

    Args:
        model (nn.Module): pytorch model 
        data_loader (torch.DataLoader): pytorch dataloader
        criterion (nn.Module): pytorch loss 
        device (str): device 

    Returns:
        Tuple[float, float]: average loss & accuracy 
    """
    model.eval()
    
    with torch.no_grad(): 
        running_loss = 0
        correct = 0

        for X, y in data_loader:
            X, y = X.to(device), y.float().to(device)   
        
            output = model(X) 
            
            # binary classification
            output = torch.sigmoid(output).squeeze()
            pred = output >= torch.FloatTensor([0.5]).to(device)
            correct += torch.sum(pred.eq(y)).item()
            running_loss += criterion(output, y).item() * X.size(0)
            
        accuracy = correct / len(data_loader.dataset)
        loss = running_loss / len(data_loader.dataset)  

        return loss, accuracy
    
    
class EarlyStopping:
    def __init__(self, patience=10, delta=0.0, mode='min', verbose=True):
        """
        Pytorch EarlyStopping

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