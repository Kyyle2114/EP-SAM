import warnings
warnings.filterwarnings('ignore')

import argparse

import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
import torchvision.transforms as tr 

from utils import seed, dataset, trainer
from cls_model import adl_fft

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--batch_size', type=int, default=64, help='batch size allocated to GPU')
    parser.add_argument('--patch_size', type=int, default=512, help='size of each WSI patch')
    parser.add_argument('--seed', type=int, default=21, help='random seed')
    parser.add_argument('--cls_model_type', type=str, default='gradcam', choices=['adl', 'gradcam'], help='Classifier model type')
    parser.add_argument('--checkpoint_dir', type=str, default='gradcam.pth', help='model checkpoint path')
    parser.add_argument('--test_image_dir', type=str, default='dataset/test/image', help='test dataset dir')
    
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

    test_set = dataset.make_dataset(
        image_dir=opts.test_image_dir,
        transform=tr.Compose([tr.Resize(opts.patch_size), tr.ToTensor()])
    )

    test_loader = DataLoader(
        test_set, 
        batch_size=opts.batch_size,
        shuffle=False
    )

    ### Model config ### 
    
    if opts.cls_model_type == 'adl':
        model = adl_fft.resnet50_adl(
            architecture_type='adl', 
            pretrained=False, 
            adl_drop_rate=0.75, 
            adl_drop_threshold=0.8
        ).to(device)
        
    # if opts.cls_model_type == 'gradcam':
    #     resnet = resnet50.ResNet50(pretrain=True).to(device)
    #     model = basic_classifier.BasicClassifier(
    #         mdoel=resnet,
    #         in_features=resnet.in_features,
    #         freezing=False,
    #         num_classes=1
    #     ).to(device)
    
    model.load_state_dict(torch.load(f'{opts.checkpoint_dir}', map_location=device))
    model.eval()
    
    for p in model.parameters():
        p.requires_grad = False

    ### loss & metric config ###  
    
    criterion = nn.BCEWithLogitsLoss()
    
    ### Evaluation phase ### 
    
    test_loss, test_acc = trainer.model_evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device
    )

    print('Test BCE Loss: %s'%test_loss)
    print('Test Accuracy: %s'%test_acc)
    print()

    return
    
if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser('Evaluating Patch Classifier', parents=[get_args_parser()])
    opts = parser.parse_args()
    
    main(opts)
    
    print('=== DONE === \n')    