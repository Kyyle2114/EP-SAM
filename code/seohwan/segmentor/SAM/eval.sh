#!/bin/bash

echo SAM_EVAL >> sam_wholebox_cam16_eval.txt

python3 eval_med.py \
    --batch_size 4 \
    --seed 21 \
    --model_type vit_b \
    --checkpoint /home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0705/code/segmentor/SAM/sam_vit_b.pth \
    --checkpoint_decoder /home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0705/code/segmentor/SAM/checkpoints/SAM_Decoder_whole_box_cam16.pth \
    --test_image_dir /home/team1/ddrive/team1/sam_dataset/camelyon16_classifier_train6000by6000/test/image \
    --test_mask_dir /home/team1/ddrive/team1/sam_dataset/camelyon16_classifier_train6000by6000/test/mask \
    >> sam_wholebox_cam16_eval.txt