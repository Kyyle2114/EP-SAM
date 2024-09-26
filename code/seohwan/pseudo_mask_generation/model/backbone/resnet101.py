import torch
import torch.nn as nn
import torchvision.models as models 
from torchvision.models import ResNet101_Weights

class ResNet101(nn.Module):
    def __init__(self, pretrain=True, progress=False):
        """
        Backbone : ResNet34
        
        returns feature map, size ([batch_size, channels, width, height])
                
        Args:
            pretrain (bool, optional): if True, use ImageNet weights(IMAGENET1K_V2). 
                                       if False, use kaiming_normal initialize in Conv layer. Defaults to True.
        """
        super(ResNet101, self).__init__()
        
        weights = ResNet101_Weights.IMAGENET1K_V2
        
        if pretrain:
            self.model = models.resnet101(weights=weights, progress=progress)
        else:
            self.model = models.resnet101(weights=None, progress=progress)
            self._initialize_weights()
            
        self.backbone = nn.Sequential(*list(self.model.children())[:-2])
        self.in_features = self.model.fc.in_features 

    def forward(self, x):
        x = self.backbone(x)
        return x 
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)