#!/bin/bash

python3 run.py \
    --batch_size 4 \
    --port 1234 \
    --dist False \
    --seed 21 \
    --model_type vit_b \
    --checkpoint sam_vit_b.pth \
    --checkpoint_cls /home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0717/code/classifier/checkpoints/internel_adl_real_lr_best.pth \
    --epoch 20 \
    --lr 2e-4 \
    --dataset_type /home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0705/code/pseudo_mask_generation/outputs_internel_mask_compare/internel_adl_cam_mask \
    --pth_name sam_internel_adl_full_cam_mask \
    --project_name sam_internel_adl_full_filter_iter_1 \
    --cuda 1 \
    
    