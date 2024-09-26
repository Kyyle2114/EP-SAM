import warnings
warnings.filterwarnings('ignore')

import argparse
import albumentations as A
import segmentation_models_pytorch as smp

import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
import torchvision.transforms as tr 

from utils import seed, dataset, trainer, iou_loss

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--batch_size', type=int, default=1, help='batch size allocated to GPU')
    parser.add_argument('--seed', type=int, default=21, help='random seed')
    parser.add_argument('--model', type=str, default='unet', help='model name to use, unet or deeplab')
    parser.add_argument('--checkpoint_dir', type=str, default='unet.pth', help='model checkpoint path')
    parser.add_argument('--test_image_dir', type=str, default='dataset/test/image', help='test dataset image dir')
    parser.add_argument('--test_mask_dir', type=str, default='dataset/test/mask', help='test dataset mask dir')
    
    return parser

def main(opts):
    """
    Model Evaluation 

    Args:
        opts (argparser): argparser
        
    After evaluation, print results 
    """
    seed.seed_everything(opts.seed)
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    
    ### dataset & dataloader ### 
    
    test_transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    test_set = dataset.make_dataset(
        image_dir=opts.test_image_dir,
        mask_dir=opts.test_mask_dir,
        transform=test_transform
    )

    test_loader = DataLoader(
        test_set, 
        batch_size=opts.batch_size,
        shuffle=False,
        pin_memory=True
    )

    ### Model config ### 
    
    if opts.model == 'unet':
        model = smp.Unet(
            encoder_name="resnet50", 
            encoder_weights=None, 
            in_channels=3, 
            classes=1, 
        )
        model.to(device)
        model.load_state_dict(torch.load(f'{opts.checkpoint_dir}', map_location=device))
        model.eval()
    
    elif opts.model == 'deeplab':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
        
        num_class = 1
        model.classifier[4] = torch.nn.Conv2d(256, num_class, kernel_size=1)
        model.aux_classifier[4] = torch.nn.Conv2d(256, num_class, kernel_size=1) 
        
        model.to(device)
        model.load_state_dict(torch.load(f'{opts.checkpoint_dir}', map_location=device))
        model.eval()

    for p in model.parameters():
        p.requires_grad = False

    ### loss & metric config ###  
    
    bceloss = nn.BCELoss().to(device)
    iouloss = iou_loss.IoULoss().to(device)
    
    ### Evaluation phase ### 
    
    test_bce_loss, test_iou_loss, test_dice, test_iou = trainer.model_evaluate(
                model=model,
                data_loader=test_loader,
                criterion=[bceloss, iouloss],
                device=device
            )
    
    print(f'\ntest_bce_loss: {test_bce_loss:.5f} \ntest_iou_loss: {test_iou_loss:.5f} \ntest_dice: {test_dice:.5f} \ntest_iou: {test_iou:.5f} \n')
    print()

    return
    
if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser('Evaluating Segmentor', parents=[get_args_parser()])
    opts = parser.parse_args()
    
    main(opts)
    
    print('=== DONE === \n')    