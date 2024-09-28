import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import DataLoader

import os
import argparse
from torchinfo import summary
import numpy as np 
import albumentations as A

from segment_anything import sam_model_registry
from segment_anything.utils import sam_trainer

from tools import seed, dataset, losses, save_weight
from patch_classifier import resnet_adl

CHECKPOINT_DIR = 'checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--batch_size', type=int, default=4, help='batch size allocated to each GPU')
    parser.add_argument('--seed', type=int, default=21, help='random seed')
    parser.add_argument('--sam_model_type', type=str, default='vit_b', help='SAM model type')
    parser.add_argument('--sam_checkpoint', type=str, default='sam_vit_b.pth', help='SAM model checkpoint')
    parser.add_argument('--epoch', type=int, default=10, help='total epoch')
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay')
    parser.add_argument('--train_dataset_dir', type=str, default='dataset/camelyon17/train', help='train dataset dir')
    
    return parser

### Fine-tuning SAM ###
def main(opts):
    """
    Model fine-tuning
    
    Returns:
        str: Save path of model checkpoint 
    """
    seed.seed_everything(opts.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    checkpoint_dir = 'checkpoints'
    file_name = 'sam_pre_decoder.pth'
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_best_path = os.path.join(checkpoint_dir, file_name)
    
    ### Dataset & Dataloader ### 

    transform = A.Compose([
        A.OneOf([
            A.HorizontalFlip(p=1),
            A.VerticalFlip(p=1),
            A.RandomRotate90(p=1),
            A.ShiftScaleRotate(p=1)
        ], p=0.5)
    ])
    
    train_set = dataset.SegmenterDataset(
        image_dir=f'{opts.train_dataset_dir}/image',
        mask_dir=f'{opts.train_dataset_dir}/initial_mask',
        transform=transform
    )
    
    val_set = dataset.SegmenterDataset(
        image_dir=f'dataset',
        mask_dir=f'dataset',
        transform=None
    )
    
    train_loader = DataLoader(
        train_set, 
        batch_size=opts.batch_size, 
        shuffle=True, 
        num_workers=opts.num_workers
    )

    val_loader = DataLoader(
        val_set, 
        batch_size=opts.batch_size, 
        shuffle=False
    )
    
    ### Model config ### 
    
    sam_checkpoint = opts.sam_checkpoint
    model_type = opts.sam_model_type

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)

    # set trainable parameters
    for _, p in sam.image_encoder.named_parameters():
        p.requires_grad = False
        
    for _, p in sam.prompt_encoder.named_parameters():
        p.requires_grad = False

    # fine-tuning mask decoder         
    for _, p in sam.mask_decoder.named_parameters():
        p.requires_grad = True
    
    # print model info 
    print()
    print('=== MODEL INFO ===')
    summary(sam)
    print()
    
    cls = resnet_adl.resnet50_adl(
        architecture_type='adl', 
        pretrained=False, 
        adl_drop_rate=0.75, 
        adl_drop_threshold=0.8
    ).to(device)
    cls.load_state_dict(torch.load('checkpoints/resnet_adl.pth', map_location=device))
    cls.eval()
        
    ### Training config ###  
   
    iouloss = losses.IoULoss()
    diceloss = losses.DiceLoss()

    es = sam_trainer.EarlyStopping(patience=10, delta=0, mode='min', verbose=True)
    optimizer = torch.optim.AdamW(sam.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=len(train_loader), 
        eta_min=0,
        last_epoch=-1
    )
    
    max_loss = np.Inf
    
    ### Training phase ###
    
    for epoch in range(opts.epoch):
        train_dice_loss, train_iou_loss, train_dice, train_iou = sam_trainer.model_train(
            model=sam,
            data_loader=train_loader,
            criterion=[diceloss, iouloss],
            optimizer=optimizer,
            device=device,
            scheduler=scheduler
        )

        val_dice_loss, val_iou_loss, val_dice, val_iou = sam_trainer.model_evaluate(
            model=sam,
            cls=cls,
            data_loader=val_loader,
            criterion=[diceloss, iouloss],
            device=device
        )
        
        val_loss = val_dice_loss + val_iou_loss
        
        # check EarlyStopping
        es(val_loss)
    
        # save best model 
        if val_loss < max_loss:
            print(f'[INFO] val_loss has been improved from {max_loss:.5f} to {val_loss:.5f}. Save model.')
            max_loss = val_loss
            _ = save_weight.save_partial_weight(model=sam, save_path=save_best_path)
        
        # print current loss & metric
        print(f'epoch {epoch+1:02d}, dice_loss: {train_dice_loss:.5f}, iou_loss: {train_iou_loss:.5f}, dice: {train_dice:.5f}, iou: {train_iou:.5f} \n')
        print(f'val_dice_loss: {val_dice_loss:.5f}, val_iou_loss: {val_iou_loss:.5f}, val_dice: {val_dice:.5f}, val_iou: {val_iou:.5f} \n')
        
        if es.early_stop:
            break    
    
    ### Evaluation phase ### 
    
    ### Generate iter 1 pseudo mask ###
    
    print(f'Model checkpoint saved at: {save_best_path} \n') 
    
    return

if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser('Preliminary Mask Decoder Fine-Tuning', parents=[get_args_parser()])
    opts = parser.parse_args()
    
    print('Preliminary Mask Decoder Fine-Tuning')
    
    main(opts)
    
    print('=== DONE === \n')  