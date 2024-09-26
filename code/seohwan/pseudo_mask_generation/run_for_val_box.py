import warnings
warnings.filterwarnings('ignore')

from segment_anything import sam_model_registry
from segment_anything.utils import *

import argparse

import torch 
from torch.utils.data import DataLoader
import torchvision.transforms as tr 

from model.backbone import resnet50
from adl import adl_fft_thres, adl_thres
from model import basic_classifier_thres
from generate_box_for_val import make_box

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--batch_size', type=int, default=2, help='batch size allocated to GPU')
    parser.add_argument('--patch_size', type=int, default=512, help='size of each WSI patch')
    parser.add_argument('--seed', type=int, default=21, help='random seed')
    parser.add_argument('--model_type', type=str, default='adl', help='Classifier model type')
    parser.add_argument('--morphology', type=str, default='False', help='morphology operation')
    parser.add_argument('--checkpoint_cls', type=str, default='adl.pth', help='Classifier model checkpoint path')
    parser.add_argument('--val_image_dir', type=str, default='dataset/val/image', help='val dataset image dir')
    parser.add_argument('--val_mask_dir', type=str, default='dataset/val/mask', help='val dataset mask dir')
    parser.add_argument('--output_dir', type=str, default='dataset/val/mask', help='output dir')
    parser.add_argument('--pmask_dir', type=str, default='dataset/train/mask', help='pseudo mask dir')
    
    return parser

def main(opts):
    """
    Model Evaluation 

    Args:
        opts (argparser): argparser
        
    After evaluation, print results 
    """
    seed.seed_everything(opts.seed)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    ### dataset & dataloader ### 

    data_set = dataset_mask.make_dataset(
        image_dir=opts.val_image_dir,
        mask_dir=opts.val_mask_dir,
        transform=tr.Resize(opts.patch_size)
    )

    data_loader = DataLoader(
        data_set, 
        batch_size=opts.batch_size,
        shuffle=False
    )

    ### Model config ### 
    
    # Classifier
    if opts.model_type == 'adl_fft':
        cls = adl_fft_thres.resnet50_adl(
            architecture_type='adl', 
            pretrained=False, 
            adl_drop_rate=0.75, 
            adl_drop_threshold=0.8
        ).to(device)
        cls.load_state_dict(torch.load(f'{opts.checkpoint_cls}', map_location=device))
        cls.eval()
        
        for p in cls.parameters():
            p.requires_grad = False
    
    elif opts.model_type == 'adl':
        cls = adl_thres.resnet50_adl(
            architecture_type='adl', 
            pretrained=False, 
            adl_drop_rate=0.75, 
            adl_drop_threshold=0.8
        ).to(device)
        cls.load_state_dict(torch.load(f'{opts.checkpoint_cls}', map_location=device))
        cls.eval()
        
        for p in cls.parameters():
            p.requires_grad = False


    # elif opts.model_type == 'gradcamplpl':
    else:
        resnet = resnet50.ResNet50(pretrain=True)

        cls = basic_classifier_thres.BasicClassifier(
            model=resnet, 
            in_features=resnet.in_features,
            freezing=True, 
            num_classes=1
        ).to(device=device)
        cls.load_state_dict(torch.load(f'{opts.checkpoint_cls}', map_location=device))
        cls.eval()
        
        for p in cls.parameters():
            p.requires_grad = False

    
    ### Mask generation phase ### 
    
    df = make_box(
        cls=cls,
        model_type=opts.model_type,
        morphology=opts.morphology,
        data_loader=data_loader,
        output_path=f'{opts.output_dir}',
        pmask_dir=opts.pmask_dir,
        device=device
    )
    print("Model type : " + opts.model_type, "morphology : " + str(opts.morphology))
    print(f"Dice : {df['dice'].mean()}")
    print(f"IoU : {df['iou'].mean()}")

    return
    
if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser('Generating Pseudo Mask', parents=[get_args_parser()])
    opts = parser.parse_args()
    
    main(opts)
    
    print('=== DONE === \n')    