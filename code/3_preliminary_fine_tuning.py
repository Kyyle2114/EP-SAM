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
    parser.add_argument('--train_image_dir', type=str, default='dataset/train/image', help='train dataset image dir')
    parser.add_argument('--val_image_dir', type=str, default='dataset/val/image', help='valid dataset image dir')
    parser.add_argument('--test_image_dir', type=str, default='dataset/test/image', help='test dataset image dir')
    
    return parser

def main(opts):
    """
    Model Training 

    Args:
        opts (argparser): argparser
        
    """
    seed.seed_everything(opts.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return
    
if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser('Generate Initial Mask & Preliminary Fine-Tuning', parents=[get_args_parser()])
    opts = parser.parse_args()
    
    print('Generate Initial Mask & Preliminary Fine-Tuning')
    
    main(opts)
    
    print('=== DONE === \n')    