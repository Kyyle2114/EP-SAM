#!/bin/bash

TRAIN_IMAGE_DIR=dataset/train/image
TRAIN_MASK_DIR=dataset/train/mask
VAL_IMAGE_DIR=dataset/val/image
VAL_MASK_DIR=dataset/val/mask

python3 run.py \
    --batch_size 4 \
    --model unet \
    --seed 21 \
    --epoch 20 \
    --lr 2e-4 \
    --weight_decay 1e-3 \
    --project_name SAM_WSSS_CAM \
    --train_image_dir $TRAIN_IMAGE_DIR \
    --train_mask_dir $TRAIN_MASK_DIR \
    --val_image_dir $VAL_IMAGE_DIR \
    --val_mask_dir $VAL_MASK_DIR \

python3 run.py \
    --batch_size 4 \
    --model deeplab \
    --seed 21 \
    --epoch 20 \
    --lr 2e-4 \
    --weight_decay 1e-3 \
    --project_name SAM_WSSS_CAM \
    --train_image_dir $TRAIN_IMAGE_DIR \
    --train_mask_dir $TRAIN_MASK_DIR \
    --val_image_dir $VAL_IMAGE_DIR \
    --val_mask_dir $VAL_MASK_DIR \