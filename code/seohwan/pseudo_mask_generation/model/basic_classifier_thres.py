import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import numpy as np
from torchvision.transforms.functional import pil_to_tensor
import cv2


class BasicClassifier(nn.Module):
    def __init__(self, model, in_features, freezing=False, num_classes=1):
        """
        Basic Classifier with Global Average Pooling
        
        Args:
            model (nn.Module): pytorch model
            in_features(int): input dimension of linear layer 
            freezing (bool, optional): if True, freeze weight of backbone. Defaults to False.
            num_classes (int, optional): number of classes. Defaults to 1(binary classification).
        """
        super(BasicClassifier, self).__init__()
        
        self.backbone = model.backbone
        
        if freezing:
            for p in self.backbone.parameters():
                p.requires_grad = False
        
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Flatten(),
                                nn.Dropout(0.3),
                                nn.Linear(in_features, num_classes))                   

    def forward(self, x):
        x = self.backbone(x)
        x = self.avg(x)
        output = self.fc(x)
        return output
    
    def _rotate(self, x: Image, angle: int) -> Image:
    
        return x.rotate(angle, expand=True)
    
    
    def _rotate_back(self, x: np.array, angle: int) -> np.array:
        
        return np.rot90(x, k=(4 - angle) // 90)
    
    
    def make_cam(
        self,
        image: Image,
        morphology: str,
        device: str,
        cam,
        targets,
        cls
    ) -> np.array:
        """
        Make CAM with rotation & morphologyEx

        Args:
            x (PIL.Image): input image 

        Returns:
            np.array: upscaled mask
        """
        
        if morphology == 'True':
            angles = [0, 90, 180, 270]
        else:
            angles = [0]

        cams = [] 

        # rotation CAM 
        for angle in angles:
            x = self._rotate(image, angle)
            x = pil_to_tensor(x).float().unsqueeze(0).div(255).to(device)
            
            rotated_cam_result = cam(input_tensor=x, targets=targets)
            rotated_cam_result = rotated_cam_result.squeeze()
            restored_cam_result = self._rotate_back(rotated_cam_result, angle)
            restored_cam_result = np.uint8(255 * restored_cam_result)
            cams.append(restored_cam_result)
            
        avg_cam = np.mean(cams, axis=0)
        avg_cam = np.uint8(avg_cam)
        
        # thresholding
        non_zero_values = avg_cam[avg_cam != 0]

        avg_cam = cv2.resize(avg_cam, (image.size[0], image.size[1])) 
        if len(non_zero_values) > 0:
            percentile_thres = np.percentile(non_zero_values, 20)
            avg_cam = np.where(avg_cam > percentile_thres, 1, 0).astype(np.uint8)
        else:
            avg_cam = np.zeros_like(avg_cam, dtype=np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
        avg_cam = cv2.morphologyEx(avg_cam, cv2.MORPH_OPEN, kernel)
        
        avg_cam[avg_cam != 0] = 1
        return avg_cam
    
    def make_bbox(self, mask: np.array) -> np.array:
        
        non_zero_points = cv2.findNonZero(mask)
        
        if non_zero_points is not None:
            x, y, w, h = cv2.boundingRect(non_zero_points)
            
            return np.array((x, y, x + w - 1, y + h - 1))
        
        else:
            
            return np.array((0, 0, 0, 0))