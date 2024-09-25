import warnings
warnings.filterwarnings('ignore')

import wandb 
import argparse
import numpy as np
import os 
from datetime import datetime

import torch 
import torch.nn as nn 
import torchvision.transforms as tr 
from torch.utils.data import DataLoader

from torchinfo import summary
from utils import seed, dataset, trainer
from cls_model import adl_fft

CHECKPOINT_DIR = 'checkpoints'
CHECKPOINT_ADL_DIR = 'checkpoints/cam17_adl_fft_lr'
CHECKPOINT_grad_DIR = 'checkpoints/gradcam'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_ADL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_grad_DIR, exist_ok=True)

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--batch_size', type=int, default=32, help='batch size allocated to GPU')
    parser.add_argument('--patch_size', type=int, default=512, help='size of each WSI patch')
    parser.add_argument('--seed', type=int, default=21, help='random seed')
    parser.add_argument('--epoch', type=int, default=50, help='total epoch')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay')
    parser.add_argument('--project_name', type=str, default='SAM_WSSS_CAM', help='WandB project name')
    parser.add_argument('--cls_model_type', type=str, default='gradcam', choices=['adl', 'gradcam'], help='Classifier model type')
    parser.add_argument('--train_image_dir', type=str, default='dataset/train/image', help='train dataset image dir')
    parser.add_argument('--val_image_dir', type=str, default='dataset/val/image', help='valid dataset image dir')
    
    return parser

def main(opts):
    """
    Model Training 

    Args:
        opts (argparser): argparser
        
    After training, print save path of model checkpoint 
    """
    seed.seed_everything(opts.seed)
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    
    ### checkpoint & WandB set ### 
    
    run_time = datetime.now()
    run_time = run_time.strftime("%b%d_%H%M%S")    
    file_name = 'cam17_adl_fft_best_lr.pth'
    save_best_path = os.path.join(CHECKPOINT_DIR, file_name)
    
    wandb.init(project=opts.project_name)
    wandb.run.name = run_time 
    
    ### dataset & dataloader ### 
    
    transform = tr.Compose(
        [
            tr.Resize(opts.patch_size), 
            tr.RandomHorizontalFlip(), 
            tr.RandomVerticalFlip(), 
            tr.RandomRotation((0, 360)),
            tr.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            tr.RandomGrayscale(p=0.1),
            tr.ToTensor()
        ]
    )

    train_set = dataset.make_dataset(
        image_dir=opts.train_image_dir,
        transform=transform
    )

    val_set = dataset.make_dataset(
        image_dir=opts.val_image_dir,
        transform=tr.Compose([tr.Resize(opts.patch_size), tr.ToTensor()])
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
    
    if opts.cls_model_type == 'adl':
        model = adl_fft.resnet50_adl(
            architecture_type='adl', 
            pretrained=True, 
            adl_drop_rate=0.75, 
            adl_drop_threshold=0.8
        ).to(device)
        
    # if opts.cls_model_type == 'gradcam':
    #     resnet = resnet50.ResNet50(pretrain=False).to(device)
    #     model = basic_classifier.BasicClassifier(
    #         model=resnet,
    #         in_features=resnet.in_features,
    #         freezing=False,
    #         num_classes=1
    #     ).to(device)
        
    for p in model.parameters():
        p.requires_grad = True
    
    print()
    print('=== MODEL INFO ===')
    summary(model)
    print()
        
    ### loss & metric config ###  
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
    es = trainer.EarlyStopping(patience=opts.epoch//2, delta=0, mode='min', verbose=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.epoch//5, gamma=0.9)    
    
    wandb.watch(
        models=model,
        criterion=criterion,
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
        train_loss, train_acc = trainer.model_train(
            model=model, 
            data_loader=train_loader, 
            criterion=criterion, 
            optimizer=optimizer, 
            device=device, 
            scheduler=scheduler, 
        )
        
        val_loss, val_acc = trainer.model_evaluate(
            model=model, 
            data_loader=val_loader, 
            criterion=criterion, 
            device=device
        )
        
        wandb.log(
            {
                'Train BCE Loss': train_loss,
                'Train Accuracy': train_acc,
                'Validation BCE Loss': val_loss,
                'Validation Accuracy': val_acc
            }, step=epoch+1
        )
        
        # Early Stop Check
        es(val_loss)

        # Save best model 
        if val_loss < max_loss:
            print(f'[INFO] val_loss has been improved from {max_loss:.5f} to {val_loss:.5f}. Save model.')
            max_loss = val_loss
            torch.save(model.state_dict(), save_best_path)
        
        # Print loss & accuracy 
        print(f'epoch {epoch+1:02d}, loss: {train_loss:.5f}, accuracy: {train_acc:.5f}, val_loss: {val_loss:.5f}, val_accuracy: {val_acc:.5f} \n')
        
        # Save all model 
        if opts.cls_model_type == 'adl':
            torch.save(model.state_dict(), f'{CHECKPOINT_ADL_DIR}/cam17_adl_fft_lr_{epoch+1}.pth')
        if opts.cls_model_type == 'gradcam':
            torch.save(model.state_dict(), f'{CHECKPOINT_grad_DIR}/ResNet_{epoch+1}.pth')
        
        if es.early_stop:
            break
    
    print(f'Model checkpoint saved at: {save_best_path} \n') 
    
    return
    
if __name__ == '__main__': 
    
    wandb.login()
    
    parser = argparse.ArgumentParser('Training Patch Classifier', parents=[get_args_parser()])
    opts = parser.parse_args()
    
    main(opts)
    
    print('=== DONE === \n')    