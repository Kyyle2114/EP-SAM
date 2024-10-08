import warnings
warnings.filterwarnings('ignore')

import argparse

import torch 
import torchvision.transforms as tr 
from torch.utils.data import DataLoader

from tools import seed, dataset, generate_initial_mask
from patch_classifier import resnet_adl

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--batch_size', type=int, default=16, help='batch size allocated to GPU')
    parser.add_argument('--patch_size', type=int, default=512, help='size of each WSI patch')
    parser.add_argument('--dataset_type', type=str, default='camelyon17', choices=['camelyon16', 'camelyon17'], help='dataset type')
    parser.add_argument('--seed', type=int, default=21, help='random seed')
    parser.add_argument('--train_image_dir', type=str, default='dataset/camelyon17/train/image', help='train dataset image dir')
    
    return parser

def main(opts):
    """
    Generate Initial Mask
    
    Args:
        opts (argparser): argparser
    """
    seed.seed_everything(opts.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    ### Dataset & Dataloader ### 
    
    train_set = dataset.InitialMaskDataset(
        image_dir=opts.train_image_dir,
        transform=tr.Resize(opts.patch_size)
    )
    
    train_loader = DataLoader(
        train_set, 
        batch_size=opts.batch_size,
        shuffle=False,
        pin_memory=True
    )

    ### Model config ### 
    
    classifier_path = f'checkpoints/{opts.dataset_type}_resnet_adl.pth'
    cls = resnet_adl.resnet50_adl(
        architecture_type='adl', 
        pretrained=False, 
        adl_drop_rate=0.75, 
        adl_drop_threshold=0.8
    ).to(device)
    cls.load_state_dict(torch.load(classifier_path, map_location=device))
    cls.eval()
    
    for p in cls.parameters():
        p.requires_grad = False
    
    ### Generate initial mask ### 
    
    generate_initial_mask.generate_initial_mask(
        classifier=cls,
        data_loader=train_loader,
        output_path=f'dataset/{opts.dataset_type}/train',
        device=device,
        dataset_type=opts.dataset_type
    )
    
    print()
    print(f'Inital masks have been generated in dataset/{opts.dataset_type}/train/initial_mask \n')
    
    return
    
if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser('Generate Initial Mask', parents=[get_args_parser()])
    opts = parser.parse_args()
    
    print('=== Generate Initial Mask ===')
    
    main(opts)
    
    print('=== DONE === \n')    