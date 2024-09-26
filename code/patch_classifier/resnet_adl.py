
"""
referenced repository : https://github.com/junsukchoe/ADL
"""
import torch
import torch.nn as nn
from torchvision.transforms.functional import pil_to_tensor

import numpy as np 
import cv2 
from PIL import Image

from .fft import extract_freq_components

__all__ = ['ADL']

class ADL(nn.Module):
    def __init__(self, adl_drop_rate=0.75, adl_drop_threshold=0.8):
        super(ADL, self).__init__()
        if not (0 <= adl_drop_rate <= 1):
            raise ValueError("Drop rate must be in range [0, 1].")
        if not (0 <= adl_drop_threshold <= 1):
            raise ValueError("Drop threshold must be in range [0, 1].")
        self.adl_drop_rate = adl_drop_rate
        self.adl_drop_threshold = adl_drop_threshold
        self.attention = None
        self.drop_mask = None

    def forward(self, input_):
        if not self.training:
            return input_
        else:
            attention = torch.mean(input_, dim=1, keepdim=True)
            importance_map = torch.sigmoid(attention)
            drop_mask = self._drop_mask(attention)
            selected_map = self._select_map(importance_map, drop_mask)
            return input_.mul(selected_map)

    def _select_map(self, importance_map, drop_mask):
        random_tensor = torch.rand([], dtype=torch.float32) + self.adl_drop_rate
        binary_tensor = random_tensor.floor()
        return (1. - binary_tensor) * importance_map + binary_tensor * drop_mask

    def _drop_mask(self, attention):
        b_size = attention.size(0)
        max_val, _ = torch.max(attention.view(b_size, -1), dim=1, keepdim=True)
        thr_val = max_val * self.adl_drop_threshold
        thr_val = thr_val.view(b_size, 1, 1, 1)
        return (attention < thr_val).float()

    def extra_repr(self):
        return 'adl_drop_rate={}, adl_drop_threshold={}'.format(
            self.adl_drop_rate, self.adl_drop_threshold)

# Define Bottleneck class
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, base_width=64):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.))
        self.conv1 = nn.Conv2d(inplanes, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# Define ADL layer position
_ADL_POSITION = [[], [], [], [0], [0, 2]]

# Define ResNetAdl class
class ResNetAdl(nn.Module):
    def __init__(self, block, layers, num_classes=1, large_feature_map=False, **kwargs):
        super(ResNetAdl, self).__init__()

        self.stride_l3 = 1 if large_feature_map else 2
        self.inplanes = 64

        self.adl_drop_rate = kwargs['adl_drop_rate']
        self.adl_threshold = kwargs['adl_drop_threshold']

        self.conv1 = nn.Conv2d(4, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, split=_ADL_POSITION[1])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, split=_ADL_POSITION[2])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=self.stride_l3, split=_ADL_POSITION[3])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, split=_ADL_POSITION[4])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        initialize_weights(self.modules(), init_mode='xavier')

    def forward(self, x, labels=None, return_cam=False):

        # Extract high-frequency components
        high_freq = extract_freq_components(x)
        high_freq_mean = high_freq.mean(dim=1, keepdim=True)
        
        X_combined = torch.cat((x, high_freq_mean), dim=1)
  
        x = self.conv1(X_combined)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        pre_logit = self.avgpool(x)
        pre_logit = pre_logit.reshape(pre_logit.size(0), -1)
        logits = self.fc(pre_logit)
        
        if return_cam:
            feature_map = x.detach().clone()
            cam_weights = self.fc.weight[labels]
            cams = (cam_weights.view(*feature_map.shape[:2], 1, 1) *
                    feature_map).mean(1, keepdim=False)
            return cams

        return logits

    def _make_layer(self, block, planes, blocks, stride, split=None):
        layers = self._layer(block, planes, blocks, stride)
        for pos in reversed(split):
            layers.insert(pos + 1, ADL(self.adl_drop_rate, self.adl_threshold))
        return nn.Sequential(*layers)

    def _layer(self, block, planes, blocks, stride):
        downsample = get_downsampling_layer(self.inplanes, block, planes, stride)
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return layers

    def _rotate(self, x: Image, angle: int) -> Image:
        
        return x.rotate(angle, expand=True)
    
    def _rotate_back(self, x: np.array, angle: int) -> np.array:
        
        return np.rot90(x, k=(4 - angle) // 90)
    
    def _normalize_minmax(self, x: np.array) -> np.array:
        
        return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8) 
    
    def make_cam(
        self,
        image: Image,
        device: str
    ) -> np.array:
        """
        Make CAM with rotation & morphologyEx

        Args:
            x (PIL.Image): input image 

        Returns:
            np.array: upscaled mask
        """
        angles = [0, 90, 180, 270]
        cams = [] 
        
        # rotation CAM 
        for angle in angles:
            x = self._rotate(image, angle)
            x = pil_to_tensor(x).float().unsqueeze(0).div(255).to(device)
            with torch.no_grad():

                # Extract high-frequency components
                high_freq = extract_freq_components(x)
                high_freq_mean = high_freq.mean(dim=1, keepdim=True)

                x_combined = torch.cat((x, high_freq_mean), dim=1)

                x = self.conv1(x_combined)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4[0](x) 
                feature_maps = x.detach().cpu().numpy() 

            # CAM 
            activation_weights = self.fc.weight[0].detach().cpu().numpy()
            cam_img = np.matmul(activation_weights, feature_maps.reshape(feature_maps.shape[1], -1)).reshape(feature_maps.shape[2:])
            
            cam_img = np.maximum(cam_img, 0)
            cam_img = cam_img - np.min(cam_img)
            cam_img = cam_img / np.max(cam_img)
            cam_img = np.uint8(255 * cam_img)
            
            cam_img = self._rotate_back(cam_img, angle)
            cams.append(cam_img)
            
        avg_cam = np.mean(cams, axis=0)
        avg_cam = np.uint8(avg_cam)

        normalized_cam = self._normalize_minmax(avg_cam)
        normalized_cam = cv2.resize(normalized_cam, (image.size[0], image.size[1]))
        normalized_cam = normalized_cam.astype(np.float32)
        
        non_zero_values = avg_cam[avg_cam != 0]
        
        if len(non_zero_values) > 0:
            percentile_thres = np.percentile(non_zero_values, 20)
            avg_cam = np.where(avg_cam > percentile_thres, 1, 0).astype(np.uint8)
            
        else:
            avg_cam = np.zeros_like(avg_cam, dtype=np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
        avg_cam = cv2.morphologyEx(avg_cam, cv2.MORPH_OPEN, kernel)
        avg_cam = cv2.resize(avg_cam, (image.size[0], image.size[1]), cv2.INTER_LANCZOS4) 

        avg_cam[avg_cam != 0] = 1
                
        return avg_cam, normalized_cam
    
def initialize_weights(modules, init_mode):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            if init_mode == 'he':
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif init_mode == 'xavier':
                nn.init.xavier_uniform_(m.weight.data)
            else:
                raise ValueError('Invalid init_mode {}'.format(init_mode))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

def align_layer(state_dict):
    keys = [key for key in sorted(state_dict.keys())]
    for key in reversed(keys):
        move = 0
        if 'layer' not in key:
            continue
        key_sp = key.split('.')
        layer_idx = int(key_sp[0][-1])
        block_idx = key_sp[1]
        if not _ADL_POSITION[layer_idx]:
            continue

        for pos in reversed(_ADL_POSITION[layer_idx]):
            if pos < int(block_idx):
                move += 1

        key_sp[1] = str(int(block_idx) + move)
        new_key = '.'.join(key_sp)
        state_dict[new_key] = state_dict.pop(key)
    return state_dict

def get_downsampling_layer(inplanes, block, planes, stride):
    outplanes = planes * block.expansion
    if stride == 1 and inplanes == outplanes:
        return
    else:
        return nn.Sequential(
            nn.Conv2d(inplanes, outplanes, 1, stride, bias=False),
            nn.BatchNorm2d(outplanes),
        )

def resnet50_adl(architecture_type, pretrained=False, pretrained_path=None, **kwargs):
    model = {'adl': ResNetAdl}[architecture_type](Bottleneck, [3, 4, 6, 3], **kwargs)
    
    return model
