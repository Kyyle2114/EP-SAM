#!/bin/bash

# python3 run_for_val_mask2.py \
#     --batch_size 1 \
#     --patch_size 512 \
#     --seed 21 \
#     --model_type adl_fft \
#     --morphology True \
#     --checkpoint_cls /home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0717/code/classifier/checkpoints/internel_adl_fft_real_lr_best.pth \
#     --val_image_dir /home/team1/ddrive/team1/sam_dataset/internal_dataset/train/image \
#     --val_mask_dir /home/team1/ddrive/team1/sam_dataset/internal_dataset/train/mask \
#     --output_dir /home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0705/code/pseudo_mask_generation/outputs_cam17_mask_compare \
#     --pmask_dir internel_resnet 

python3 run_for_val_mask2.py \
    --batch_size 1 \
    --patch_size 512 \
    --seed 21 \
    --model_type gradcam \
    --morphology True \
    --checkpoint_cls /home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0717/code/classifier/checkpoints/internel_resnet_real_lr_best.pth \
    --val_image_dir /home/team1/ddrive/team1/sam_dataset/internal_dataset/train/image \
    --val_mask_dir /home/team1/ddrive/team1/sam_dataset/internal_dataset/train/mask \
    --output_dir /home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0705/code/pseudo_mask_generation/outputs_cam17_mask_compare \
    --pmask_dir internel_resnet 


# adl_fft_best :/home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0717/code/classifier/checkpoints/cam17_adl_fft_lr_1e5_lr_best.pth
# adl best : /home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0717/code/classifier/checkpoints/cam17_adl_best.pth
# resnet best : //home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0717/code/classifier/checkpoints/cam17_gracam_best.pth