import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

class CAM:
   """
   CAM class
   
   Args:
       model (nn.Module): CAM 계산에 사용할 모델
       device (torch.device): cuda
       preprocess_fn (callable): 이미지 전처리 함수 ex. torch.transform
   """
   def __init__(self, model, device, preprocess_fn):
       self.model = model
       self.device = device
       self.preprocess_fn = preprocess_fn

   def show_rotated_images(self, image):
       """
       입력 이미지를 0, 90, 180, 270도 회전한 이미지 리스트를 반환
       
       Args:
           image (PIL.Image.Image): input image
           
       Returns:
           list: 회전된 이미지 리스트
       """
       images = []
       np_image = np.array(image)  # PIL -> NumPy

       # 원본 이미지 추가
       images.append(Image.fromarray(np_image))

       # 90, 180, 270도 rotation
       for _ in range(3):
           np_image = np.rot90(np_image)
           images.append(Image.fromarray(np_image))

       return images

   def show_rotation_CAM(self, img):
       """
       입력 이미지와 회전된 이미지들에 대한 CAM 계산
       
       Args:
           img (PIL.Image.Image): 입력 이미지
           
       Returns:
           tuple: (평균 CAM 이미지, 평균 예측 클래스 ID)
       """
       # 회전된 이미지들에 대한 CAM 계산
       cam_images = []
       class_ids = []
       images = self.show_rotated_images(img)

       for i, rotated_img in enumerate(images):
           img_input = self.preprocess_fn(rotated_img).unsqueeze(0).to(self.device)
           output = self.model(img_input)
           class_id = int(output.argmax().item())
           feature_maps = self.model.backbone(img_input).squeeze().detach().cpu().numpy()
           activation_weights = list(self.model.fc[:2].parameters())[-2].data.cpu().numpy()

           # CAM 이미지 생성
           cam_img = np.matmul(activation_weights[class_id], feature_maps.reshape(feature_maps.shape[0], -1)).reshape(feature_maps.shape[1:])
           cam_img = np.maximum(cam_img, 0)
           cam_img = cam_img - np.min(cam_img)
           cam_img = cam_img / np.max(cam_img)
           cam_img = np.uint8(255 * cam_img)

           # Heat Map 생성 및 합성
           numpy_img = np.asarray(rotated_img)
           heatmap = cv2.applyColorMap(cv2.resize(255 - cam_img, (numpy_img.shape[1], numpy_img.shape[0])), cv2.COLORMAP_JET)
           result = numpy_img * 0.5 + heatmap * 0.3
           result = np.uint8(result)

           cam_images.append(np.rot90(result, -i))
           class_ids.append(class_id)

       # CAM 이미지들의 평균 계산
       avg_cam_img = np.mean(cam_images, axis=0).astype(np.uint8)
       avg_class_id = np.bincount(class_ids).argmax()
       
       return avg_cam_img, avg_class_id
    
    # CAM 시각화 함수
   def show_CAM(self, img):
        # 이미지 전처리
        img_input = preprocess(img).unsqueeze(0).to(device)x
    
        # 모델 출력 및 CAM 계산
        output = model(img_input)
        
        class_id = int(output.argmax().item())
        feature_maps = model.backbone(img_input).squeeze().detach().cpu().numpy()
    
        activation_weights = list(model.fc[:2].parameters())[-2].data.cpu().numpy()
    
        # CAM 이미지 생성
        cam_img = np.matmul(activation_weights[class_id], feature_maps.reshape(feature_maps.shape[0], -1)).reshape(feature_maps.shape[1:])
        cam_img = np.maximum(cam_img, 0)
        cam_img = cam_img - np.min(cam_img)
        cam_img = cam_img / np.max(cam_img)
        cam_img = np.uint8(255 * cam_img)
    
        # Heat Map 생성 및 합성
        numpy_img = np.asarray(img)
        heatmap = cv2.applyColorMap(cv2.resize(255 - cam_img, (numpy_img.shape[1], numpy_img.shape[0])), cv2.COLORMAP_JET)
        result = numpy_img * 0.5 + heatmap * 0.3
        result = np.uint8(result)

        return result, class_id
        
   def show_images_with_cam(self, images, labels, num_images, use_rotation=False):
       """
       입력 이미지들에 대한 CAM 계산 및 시각화
       
       Args:
           images (list): 입력 이미지 배치
           labels (list): 입력 이미지 레이블
           num_images (int): 시각화할 이미지 수
           use_rotation (bool): rotation 유무에 따른 시각화, default=false
       """
       rows = num_images // 4 + (num_images % 4 > 0)
       fig, axes = plt.subplots(rows, 4, figsize=(20, rows * 5))

       for i in range(num_images):
           row = i // 4
           col = i % 4
           img = Image.fromarray((images[i] * 255).astype(np.uint8))

           # rotation 여부
           if use_rotation:
               result, predicted_class_id = self.show_rotation_CAM(img)
           else:
               result, predicted_class_id = self.show_CAM(img)

           title = f'GT: {labels[i]}, Pred: {predicted_class_id}'

           if rows > 1:
               axes[row, col].imshow(result)
               axes[row, col].set_title(title)
               axes[row, col].axis('off')
           else:
               axes[col].imshow(result)
               axes[col].set_title(title)
               axes[col].axis('off')

       for i in range(num_images, rows * 4):
           row = i // 4
           col = i % 4
           if rows > 1:
               axes[row, col].axis('off')
           else:
               axes[col].axis('off')

       plt.show()