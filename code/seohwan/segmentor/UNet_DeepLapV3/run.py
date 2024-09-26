import warnings
warnings.filterwarnings('ignore')

import wandb 
import argparse
import numpy as np
import os 
from datetime import datetime
import albumentations as A
from torchinfo import summary
import segmentation_models_pytorch as smp

import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader

from utils import seed, dataset, trainer, iou_loss

CHECKPOINT_DIR = 'checkpoints'
CHECKPOINT_ALL_DIR = 'checkpoints/all'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_ALL_DIR, exist_ok=True)

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--batch_size', type=int, default=4, help='batch size allocated to GPU')
    parser.add_argument('--model', type=str, default='unet', help='model name to use, unet or deeplab')
    parser.add_argument('--seed', type=int, default=21, help='random seed')
    parser.add_argument('--epoch', type=int, default=50, help='total epoch')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay')
    parser.add_argument('--project_name', type=str, default='SAM_WSSS_CAM', help='WandB project name')
    parser.add_argument('--train_image_dir', type=str, default='dataset/train/image', help='train dataset image dir')
    parser.add_argument('--train_mask_dir', type=str, default='dataset/train/mask', help='train dataset mask dir')
    parser.add_argument('--val_image_dir', type=str, default='dataset/val/image', help='valid dataset image dir')
    parser.add_argument('--val_mask_dir', type=str, default='dataset/val/mask', help='valid dataset mask dir')
    
    return parser

def main(opts):
    """
    Model Training 

    Args:
        opts (argparser): argparser
        
    After training, print save path of model checkpoint 
    """
    seed.seed_everything(opts.seed)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    ### checkpoint & WandB set ### 
    
    run_time = datetime.now()
    run_time = run_time.strftime("%b%d_%H%M%S")    
    file_name = run_time + '_' + opts.model + '.pth'
    save_best_path = os.path.join(CHECKPOINT_DIR, file_name)
    
    wandb.init(project=opts.project_name)
    wandb.run.name = run_time 
    
    ### dataset & dataloader ### 
    
    train_transform = A.Compose([
        A.OneOf([
                A.HorizontalFlip(p=1),
                A.VerticalFlip(p=1),
                A.RandomRotate90(p=1),
                A.ShiftScaleRotate(p=1)
            ], p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    val_transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    train_set = dataset.make_dataset(
        image_dir=opts.train_image_dir,
        mask_dir=opts.train_mask_dir,
        transform=train_transform
    )
    
    val_set = dataset.make_dataset(
        image_dir=opts.val_image_dir,
        mask_dir=opts.val_mask_dir,
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_set, 
        batch_size=opts.batch_size,
        shuffle=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set, 
        batch_size=opts.batch_size,
        shuffle=False,
        pin_memory=True,
    )

    ### Model config ### 
    
    if opts.model == 'unet':
        model = smp.Unet(
            encoder_name="resnet50", 
            encoder_weights="imagenet", 
            in_channels=3, 
            classes=1, 
        )
        model.to(device)
    
    elif opts.model == 'deeplab':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
        
        num_class = 1
        model.classifier[4] = torch.nn.Conv2d(256, num_class, kernel_size=1)
        model.aux_classifier[4] = torch.nn.Conv2d(256, num_class, kernel_size=1) 
        
        model.to(device)
    
    for p in model.parameters():
        p.requires_grad = True
    
    print()
    print('=== MODEL INFO ===')
    summary(model)
    print()
        
    ### loss & metric config ###  

    bceloss = nn.BCELoss().to(device)
    iouloss = iou_loss.IoULoss().to(device)
    
    # EarlyStopping : Determined based on the validation loss. Lower is better(mode='min').
    es = trainer.EarlyStopping(patience=opts.epoch//2, delta=0, mode='min', verbose=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=opts.lr, 
        weight_decay=opts.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=len(train_loader), 
        eta_min=0,
        last_epoch=-1
    )
    
    wandb.watch(
        models=model,
        criterion=(bceloss, iouloss),
        log='all',
        log_freq=10
    )

    wandb.run.summary['optimizer'] = type(optimizer).__name__
    wandb.run.summary['scheduler'] = type(scheduler).__name__
    wandb.run.summary['initial lr'] = opts.lr
    wandb.run.summary['weight decay'] = opts.weight_decay 
    wandb.run.summary['total epoch'] = opts.epoch
    
    ### Training phase ### 
    
    max_loss = np.inf
    
    for epoch in range(opts.epoch):
        train_bce_loss, train_iou_loss, train_dice, train_iou = trainer.model_train(
            model=model,
            data_loader=train_loader,
            criterion=[bceloss, iouloss],
            optimizer=optimizer,
            device=device,
            scheduler=scheduler
        )
        
        val_bce_loss, val_iou_loss, val_dice, val_iou = trainer.model_evaluate(
            model=model,
            data_loader=val_loader,
            criterion=[bceloss, iouloss],
            device=device
        )
        
        val_loss = val_bce_loss + val_iou_loss
        
        wandb.log(
            {
                'Train BCE Loss': train_bce_loss,
                'Train IoU Loss': train_iou_loss,
                'Train Dice Metric': train_dice,
                'Train IoU Metric': train_iou
            }, step=epoch+1
        )
    
        wandb.log(
            {
                'Validation BCE Loss': val_bce_loss,
                'Validation IoU Loss': val_iou_loss,
                'Validation Dice Metric': val_dice,
                'Validation IoU Metric': val_iou
            }, step=epoch+1
        )
        
        # Early Stop Check
        es(val_loss)

        # Save best model 
        if val_loss < max_loss:
            print(f'[INFO] val_loss has been improved from {max_loss:.5f} to {val_loss:.5f}. Save model.')
            max_loss = val_loss
            torch.save(model.state_dict(), save_best_path)
        
        # print current loss / metric 
        print(f'epoch {epoch+1:02d}, bce_loss: {train_bce_loss:.5f}, iou_loss: {train_iou_loss:.5f}, dice: {train_dice:.5f}, iou: {train_iou:.5f},', end=' ')
        print(f'val_bce_loss: {val_bce_loss:.5f}, val_iou_loss: {val_iou_loss:.5f}, val_dice: {val_dice:.5f}, val_iou: {val_iou:.5f} \n')
        
        # Save all model 
        torch.save(model.state_dict(), f'{CHECKPOINT_ALL_DIR}/{opts.model}_{epoch+1}.pth')
        
        if es.early_stop:
            break
    
    print(f'Model checkpoint saved at: {save_best_path} \n') 
    
    return
    
if __name__ == '__main__': 
    
    wandb.login()
    
    parser = argparse.ArgumentParser('Training Segmentor', parents=[get_args_parser()])
    opts = parser.parse_args()
    
    main(opts)
    
    print('=== DONE === \n')    