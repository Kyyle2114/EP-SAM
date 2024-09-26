import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights  # ResNet50에 적합한 가중치 클래스를 임포트

class ResNet50(nn.Module):
    def __init__(self, pretrain=True, progress=False):
        """
        Backbone : ResNet50
        
        returns feature map, size ([batch_size, channels, width, height])
                
        Args:
            pretrain (bool, optional): if True, use ImageNet weights (IMAGENET1K_V2). 
                                       if False, use kaiming_normal initialize in Conv layer. Defaults to True.
        """
        super(ResNet50, self).__init__()
        
        weights = ResNet50_Weights.IMAGENET1K_V2  # ResNet50에 맞는 가중치 선택
        
        if pretrain:
            self.model = models.resnet50(weights=weights, progress=progress)
        else:
            self.model = models.resnet50(weights=None, progress=progress)
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
