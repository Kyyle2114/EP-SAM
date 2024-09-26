#!/bin/bash

# python3 run_for_val_box.py \
#     --batch_size 1 \
#     --patch_size 512 \
#     --seed 21 \
#     --model_type adl \
#     --morphology False \
#     --checkpoint_cls /home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0717/code/classifier/checkpoints/cam17_adl_best.pth \
#     --val_image_dir /home/team1/ddrive/team1/sam_dataset/camelyon17_classifier_train6000by6000/val/image \
#     --val_mask_dir /home/team1/ddrive/team1/sam_dataset/camelyon17_classifier_train6000by6000/val/mask \
#     --output_dir /home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0705/code/pseudo_mask_generation/outputs_cam17_bbox \
#     --pmask_dir cam17_adl_no_morphology

# python3 run_for_val_box.py \
#     --batch_size 1 \
#     --patch_size 512 \
#     --seed 21 \
#     --model_type adl \
#     --morphology True \
#     --checkpoint_cls /home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0717/code/classifier/checkpoints/cam17_adl_best.pth \
#     --val_image_dir /home/team1/ddrive/team1/sam_dataset/camelyon17_classifier_train6000by6000/val/image \
#     --val_mask_dir /home/team1/ddrive/team1/sam_dataset/camelyon17_classifier_train6000by6000/val/mask \
#     --output_dir /home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0705/code/pseudo_mask_generation/outputs_cam17_bbox \
#     --pmask_dir cam17_adl_morphology

# python3 run_for_val_box.py \
#     --batch_size 1 \
#     --patch_size 512 \
#     --seed 21 \
#     --model_type adl_fft \
#     --morphology False \
#     --checkpoint_cls /home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0717/code/classifier/checkpoints/cam17_adl_fft_best.pth \
#     --val_image_dir /home/team1/ddrive/team1/sam_dataset/camelyon17_classifier_train6000by6000/val/image \
#     --val_mask_dir /home/team1/ddrive/team1/sam_dataset/camelyon17_classifier_train6000by6000/val/mask \
#     --output_dir /home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0705/code/pseudo_mask_generation/outputs_cam17_bbox \
#     --pmask_dir cam17_adl_fft_no_morphology

# python3 run_for_val_box.py \
#     --batch_size 1 \
#     --patch_size 512 \
#     --seed 21 \
#     --model_type adl_fft \
#     --morphology True \
#     --checkpoint_cls /home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0717/code/classifier/checkpoints/cam17_adl_fft_best.pth \
#     --val_image_dir /home/team1/ddrive/team1/sam_dataset/camelyon17_classifier_train6000by6000/val/image \
#     --val_mask_dir /home/team1/ddrive/team1/sam_dataset/camelyon17_classifier_train6000by6000/val/mask \
#     --output_dir /home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0705/code/pseudo_mask_generation/outputs_cam17_bbox \
#     --pmask_dir cam17_adl_fft_morphology

# python3 run_for_val_box.py \
#     --batch_size 1 \
#     --patch_size 512 \
#     --seed 21 \
#     --model_type adl_fft \
#     --morphology False \
#     --checkpoint_cls /home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0717/code/classifier/checkpoints/cam17_adl_fft_best.pth \
#     --val_image_dir /home/team1/ddrive/team1/sam_dataset/camelyon17_classifier_train6000by6000/val/image \
#     --val_mask_dir /home/team1/ddrive/team1/sam_dataset/camelyon17_classifier_train6000by6000/val/mask \
#     --output_dir /home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0705/code/pseudo_mask_generation/outputs_cam17_bbox \
#     --pmask_dir cam17_adl_fft

python3 run_for_val_box.py \
    --batch_size 1 \
    --patch_size 512 \
    --seed 21 \
    --model_type gradcam \
    --morphology True \
    --checkpoint_cls /home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0717/code/classifier/checkpoints/internel_resnet_real_lr_best.pth \
    --val_image_dir /home/team1/ddrive/team1/sam_dataset/internal_dataset/test/image \
    --val_mask_dir /home/team1/ddrive/team1/sam_dataset/internal_dataset/test/mask \
    --output_dir /home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0705/code/pseudo_mask_generation/outputs_cam17_bbox \
    --pmask_dir internel_resnet   