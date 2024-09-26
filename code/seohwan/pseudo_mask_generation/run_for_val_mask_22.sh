#!/bin/bash

python3 run_for_val_mask_data_custom_2.py \
    --batch_size 1 \
    --patch_size 512 \
    --seed 21 \
    --model_type adl_fft \
    --morphology True \
    --checkpoint_cls /home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0717/code/classifier/checkpoints/cam17_adl_fft_3_best.pth \
    --val_image_dir /home/team1/ddrive/team1/sam_dataset/camelyon17_classifier_train6000by6000/train/image \
    --val_mask_dir /home/team1/ddrive/team1/sam_dataset/camelyon17_classifier_train6000by6000/train/mask \
    --output_dir /home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0705/code/pseudo_mask_generation/outputs_cam17_mask_compare \
    --pmask_dir cam17_adl_fft_cam_mask \
    --dataset_type cam17 \
    --test_or_train train

# /home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0717/code/classifier/checkpoints/internel_adl_fft_real_lr_best.pth
# /home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0717/code/classifier/checkpoints/internel_adl_best.pth

# python3 run_for_train_point_softmax.py \
#     --batch_size 1 \
#     --patch_size 512 \
#     --seed 21 \
#     --model_type vit_b \
#     --checkpoint_sam sam_vit_b.pth \
#     --checkpoint_cls /home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0717/code/classifier/checkpoints/internel_resnet_real_lr_best.pth \
#     --train_image_dir /home/team1/ddrive/team1/sam_dataset/internal_dataset/train/image \
#     --train_mask_dir /home/team1/ddrive/team1/sam_dataset/internal_dataset/train/mask \
#     --pmask_dir train_point_prompt_pseudo_masks_test

# python3 run_for_val_mask_data_custom.py \
#     --batch_size 1 \
#     --patch_size 512 \
#     --seed 21 \
#     --model_type layercam \
#     --morphology True \
#     --checkpoint_cls /home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0717/code/classifier/checkpoints/cam16_resnet_best.pth \
#     --val_image_dir /home/team1/ddrive/team1/sam_dataset/camelyon16_classifier_train6000by6000/test/image \
#     --val_mask_dir /home/team1/ddrive/team1/sam_dataset/camelyon16_classifier_train6000by6000/test/mask \
#     --output_dir /home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0705/code/pseudo_mask_generation/outputs_cam16_mask_compare \
#     --pmask_dir cam16_resnet \
#     --dataset_type cam16 \
#     --test_or_train test

# python3 run_for_val_mask_data_custom.py \
#     --batch_size 1 \
#     --patch_size 512 \
#     --seed 21 \
#     --model_type eigencam \
#     --morphology True \
#     --checkpoint_cls /home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0717/code/classifier/checkpoints/cam17_gracam_best.pth \
#     --val_image_dir /home/team1/ddrive/team1/sam_dataset/camelyon17_classifier_train6000by6000/train/image \
#     --val_mask_dir /home/team1/ddrive/team1/sam_dataset/camelyon17_classifier_train6000by6000/train/mask \
#     --output_dir /home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0705/code/pseudo_mask_generation/outputs_cam17_mask_compare \
#     --pmask_dir cam17_resnet \
#     --dataset_type cam17 \
#     --test_or_train train

# python3 run_for_val_mask_data_custom.py \
#     --batch_size 1 \
#     --patch_size 512 \
#     --seed 21 \
#     --model_type eigencam \
#     --morphology True \
#     --checkpoint_cls /home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0717/code/classifier/checkpoints/internel_resnet_real_lr_best.pth \
#     --val_image_dir /home/team1/ddrive/team1/sam_dataset/internal_dataset/train/image \
#     --val_mask_dir /home/team1/ddrive/team1/sam_dataset/internal_dataset/train/image \
#     --output_dir /home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0705/code/pseudo_mask_generation/outputs_internel_mask_compare \
#     --pmask_dir internel_resnet \
#     --dataset_type internel \
#     --test_or_train train

# adl_fft_best :/home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0717/code/classifier/checkpoints/internel_adl_fft_real/internel_adl_fft_real_14.pth
# adl best : /home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0717/code/classifier/checkpoints/internel_adl_real_lr_best.pth
# resnet best : //home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0717/code/classifier/checkpoints/cam17_gracam_best.pth