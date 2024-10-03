import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import DataLoader

import argparse

from segment_anything import sam_model_registry
from segment_anything.utils import sam_trainer

from tools import seed, dataset, losses, save_weight
from patch_classifier import resnet_adl

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--batch_size', type=int, default=4, help='batch size allocated to each GPU')
    parser.add_argument('--seed', type=int, default=21, help='random seed')
    parser.add_argument('--sam_model_type', type=str, default='vit_b', help='SAM model type')
    parser.add_argument('--sam_checkpoint', type=str, default='sam_vit_b.pth', help='SAM model checkpoint')
    parser.add_argument('--resnet_checkpoint', type=str, default='checkpoints/camelyon17_resnet_adl.pth', help='classifier model checkpoint')
    parser.add_argument('--sam_decoder_checkpoint', type=str, default='checkpoints/camelyon17_decoder_iter3.pth', help='SAM mask decoder checkpoint')
    parser.add_argument('--test_dataset_dir', type=str, default='dataset/camelyon17/test', help='test dataset dir')
    
    return parser

def main(opts):
    """
    Model Inference

    Args:
        opts (argparser): argparser
    """
    seed.seed_everything(opts.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ### Dataset & Dataloader ### 

    test_set = dataset.SegmenterDataset(
        image_dir=f'{opts.test_dataset_dir}/image',
        mask_dir=f'{opts.test_dataset_dir}/mask',
        transform=None
    )
    
    test_loader = DataLoader(
        test_set, 
        batch_size=opts.batch_size, 
        shuffle=False
    )
    
    ### Model config ### 
    
    sam_checkpoint = opts.sam_checkpoint
    model_type = opts.sam_model_type

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    
    sam = save_weight.load_partial_weight(
            model=sam,
            load_path=opts.sam_decoder_checkpoint
        )
    sam.eval()
    
    cls = resnet_adl.resnet50_adl(
        architecture_type='adl', 
        pretrained=False, 
        adl_drop_rate=0.75, 
        adl_drop_threshold=0.8
    ).to(device)
    cls.load_state_dict(torch.load(opts.resnet_checkpoint, map_location=device))
    cls.eval()
           
    iouloss = losses.IoULoss()
    diceloss = losses.DiceLoss()
    
    ### Model inference ###  
        
    test_dice_loss, test_iou_loss, test_dice, test_iou = sam_trainer.model_evaluate(
        model=sam,
        classifier=cls,
        data_loader=test_loader,
        criterion=[diceloss, iouloss],
        device=device
    )
    
    print('Test Dice Loss: %s'%test_dice_loss)
    print('Test IoU Loss: %s'%test_iou_loss)
    print('Test Dice: %s'%test_dice)
    print('Test IoU: %s'%test_iou)
    print()
    
    return
    
if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser('Model Inference', parents=[get_args_parser()])
    opts = parser.parse_args()
    
    print('=== Model Inference ===')
    
    print(f'Dataset path: {opts.test_dataset_dir}')
    main(opts)
    
    print('=== DONE === \n')    