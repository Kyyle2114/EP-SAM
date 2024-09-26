#!/bin/bash


python3 run_for_pseudo_mask.py \
    --batch_size 1 \
    --patch_size 512 \
    --seed 21 \
    --model_type vit_b \
    --checkpoint_sam sam_vit_b.pth \
    --checkpoint_decoder /home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0705/code/segmentor/SAM/checkpoints/sam_internel_cam_full_filter_iter_1_dice_iou_lr_5e-6_point_prompt_proba_real.pth \
    --train_image_dir /home/team1/ddrive/team1/sam_dataset/internal_dataset/test/image \
    --train_mask_dir /home/team1/ddrive/team1/sam_dataset/internal_dataset/test/mask \
    --pmask_dir train_pseudo_masks_point_prompt_softmax_adl_internel_test \
    --checkpoint_cls /home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0717/code/classifier/checkpoints/internel_adl_fft_real_lr_best.pth \
    --cam_type adl_fft