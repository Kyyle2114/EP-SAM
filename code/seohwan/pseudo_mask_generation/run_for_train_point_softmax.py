import warnings
warnings.filterwarnings('ignore')

from segment_anything import sam_model_registry
from segment_anything.utils import *

import argparse

import torch 
from torch.utils.data import DataLoader
import torchvision.transforms as tr
from model.backbone import resnet50

from adl import adl_fft_internel_point_softmax, adl_internel_point_softmax
from generate_mask_for_train_fft_point_softmax import make_point_mask
from mask_compare_point import compare_mask
from model import basic_classifier_thres_2

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--batch_size', type=int, default=2, help='batch size allocated to GPU')
    parser.add_argument('--patch_size', type=int, default=512, help='size of each WSI patch')
    parser.add_argument('--seed', type=int, default=21, help='random seed')
    parser.add_argument('--model_type', type=str, default='vit_h', help='SAM model type')
    parser.add_argument('--checkpoint_sam', type=str, default='sam_vit_h.pth', help='SAM model checkpoint path')
    parser.add_argument('--checkpoint_decoder', type=str, default='sam_vit_h.pth', help='SAM model decoder checkpoint path')
    parser.add_argument('--checkpoint_cls', type=str, default='adl.pth', help='Classifier model checkpoint path')
    parser.add_argument('--train_image_dir', type=str, default='dataset/train/image', help='train dataset image dir')
    parser.add_argument('--train_mask_dir', type=str, default='dataset/train/mask', help='train dataset mask dir')
    parser.add_argument('--pmask_dir', type=str, default='dataset/train/mask', help='pseudo mask dir')
    parser.add_argument('--cam_type', type=str, default='dataset/train/mask', help='pseudo mask dir')
    
    
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

    data_set = dataset_mask.make_dataset(
        image_dir=opts.train_image_dir,
        mask_dir=opts.train_mask_dir,
        transform=tr.Resize((opts.patch_size, opts.patch_size))
    )

    data_loader = DataLoader(
        data_set, 
        batch_size=opts.batch_size,
        shuffle=False
    )

    ### Model config ### 
    
    if opts.cam_type == 'adl_fft':
        # Classifier
        cls = adl_fft_internel_point_softmax.resnet50_adl(
            architecture_type='adl', 
            pretrained=False, 
            adl_drop_rate=0.75, 
            adl_drop_threshold=0.8
        ).to(device)
        cls.load_state_dict(torch.load(f'{opts.checkpoint_cls}', map_location=device))
        cls.eval()
        
        for p in cls.parameters():
            p.requires_grad = False
            
    elif opts.cam_type == 'adl':
        cls = adl_internel_point_softmax.resnet50_adl(
            architecture_type='adl', 
            pretrained=False, 
            adl_drop_rate=0.75, 
            adl_drop_threshold=0.8
        ).to(device)
        cls.load_state_dict(torch.load(f'{opts.checkpoint_cls}', map_location=device))
        cls.eval()
        
    else:
        resnet = resnet50.ResNet50(pretrain=False)

        cls = basic_classifier_thres_2.BasicClassifier(
            model=resnet, 
            in_features=resnet.in_features,
            freezing=True, 
            num_classes=1
        ).to(device=device)
        cls.load_state_dict(torch.load(f'{opts.checkpoint_cls}', map_location=device))
        cls.eval()
        
        for p in cls.parameters():
            p.requires_grad = False
        
    # SAM 
    sam_checkpoint = opts.checkpoint_sam
    model_type = opts.model_type

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    sam.eval()
    
    sam = save_weight.load_partial_weight(
        model=sam,
        load_path=opts.checkpoint_decoder,
        dist=False
    )
    
    for p in sam.parameters():
        p.requires_grad = False 
    
    ### Mask generation phase ### 
    
    df = make_point_mask(
        sam=sam,
        cls=cls,
        data_loader=data_loader,
        output_path='outputs_vit_b_internel_adl_cam',
        pmask_dir=opts.pmask_dir,
        device=device,
        cam_type=opts.cam_type
    )
    
    print(df.head())
    
    compare_mask(
        mask_dir=opts.train_mask_dir,
        pmask_dir=f'outputs_vit_b_internel_adl_cam/{opts.pmask_dir}',
    )

    return
    
if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser('Generating Pseudo Mask', parents=[get_args_parser()])
    opts = parser.parse_args()
    
    main(opts)
    
    print('=== DONE === \n')    