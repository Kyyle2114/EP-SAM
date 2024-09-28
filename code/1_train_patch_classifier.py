import warnings
warnings.filterwarnings('ignore')

import argparse
import numpy as np
import os 

import torch 
import torch.nn as nn 
import torchvision.transforms as tr 
from torch.utils.data import DataLoader

from torchinfo import summary
from tools import seed, dataset, classifier_trainer
from patch_classifier import resnet_adl

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--batch_size', type=int, default=16, help='batch size allocated to GPU')
    parser.add_argument('--patch_size', type=int, default=512, help='size of each WSI patch')
    parser.add_argument('--seed', type=int, default=22, help='random seed')
    parser.add_argument('--epoch', type=int, default=50, help='total epoch')
    parser.add_argument('--lr', type=float, default=1e-5, help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay')
    parser.add_argument('--train_image_dir', type=str, default='dataset/camelyon17/train/image', help='train dataset image dir')
    parser.add_argument('--val_image_dir', type=str, default='dataset/camelyon17/val/image', help='valid dataset image dir')
    parser.add_argument('--test_image_dir', type=str, default='dataset/camelyon17/test/image', help='test dataset image dir')
    
    return parser

def main(opts):
    """
    Model Training 

    Args:
        opts (argparser): argparser
        
    """
    seed.seed_everything(opts.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    checkpoint_dir = 'checkpoints'
    file_name = 'resnet_adl.pth'
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_best_path = os.path.join(checkpoint_dir, file_name)
    
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

    train_set = dataset.ClassifierDataset(
        image_dir=opts.train_image_dir,
        transform=transform
    )

    val_set = dataset.ClassifierDataset(
        image_dir=opts.val_image_dir,
        transform=tr.Compose([tr.Resize(opts.patch_size), tr.ToTensor()])
    )
    
    test_set = dataset.ClassifierDataset(
        image_dir=opts.test_image_dir,
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
    
    test_loader = DataLoader(
        test_set, 
        batch_size=opts.batch_size,
        shuffle=False,
        pin_memory=True,
    )

    ### Model config ### 
    
    model = resnet_adl.resnet50_adl(
        architecture_type='adl', 
        pretrained=True, 
        adl_drop_rate=0.75, 
        adl_drop_threshold=0.8
    ).to(device)
        
    for p in model.parameters():
        p.requires_grad = True
    
    print()
    print('=== MODEL INFO ===')
    summary(model)
    print()
    
    ### loss & metric config ###  
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
    es = classifier_trainer.EarlyStopping(patience=opts.epoch//2, delta=0, mode='min', verbose=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.epoch//5, gamma=0.9)    
    
    ### Training phase ### 
    
    max_loss = np.inf
    
    for epoch in range(opts.epoch):
        train_loss, train_acc = classifier_trainer.model_train(
            model=model, 
            data_loader=train_loader, 
            criterion=criterion, 
            optimizer=optimizer, 
            device=device, 
            scheduler=scheduler, 
        )
        
        val_loss, val_acc = classifier_trainer.model_evaluate(
            model=model, 
            data_loader=val_loader, 
            criterion=criterion, 
            device=device
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
        
        if es.early_stop:
            break
        
    ### Evaluation phase ### 
    
    model.load_state_dict(torch.load(save_best_path, map_location=device))
    model.eval()
    
    for p in model.parameters():
        p.requires_grad = False
        
    test_loss, test_acc = classifier_trainer.model_evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device
    )

    print('Test BCE Loss: %s'%test_loss)
    print('Test Accuracy: %s'%test_acc)
    print()
    
    print(f'Model checkpoint saved at: {save_best_path} \n') 
    
    return
    
if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser('Training Patch Classifier', parents=[get_args_parser()])
    opts = parser.parse_args()
    
    print('=== Training Patch Classifier ===')
    
    main(opts)
    
    print('=== DONE === \n')    