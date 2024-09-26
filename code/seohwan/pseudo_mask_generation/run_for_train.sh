#!/bin/bash

# python3 run_for_train.py \
#     --batch_size 1 \
#     --patch_size 512 \
#     --seed 21 \
#     --model_type vit_b \
#     --checkpoint_sam /home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0705/code/segmentor/SAM/medsam_vit_b.pth \
#     --checkpoint_cls /home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0717/code/classifier/checkpoints/cam17_adl_best.pth \
#     --train_image_dir /home/team1/ddrive/team1/sam_dataset/camelyon17_classifier_train6000by6000/train/image \
#     --train_mask_dir /home/team1/ddrive/team1/sam_dataset/camelyon17_classifier_train6000by6000/train/mask

#!/bin/bash

# python3 run_for_train_point.py \
#     --batch_size 1 \
#     --patch_size 512 \
#     --seed 21 \
#     --model_type vit_b \
#     --checkpoint_sam sam_vit_b.pth \
#     --checkpoint_cls /home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0717/code/classifier/checkpoints/Internel_adl_fft_best.pth \
#     --train_image_dir /home/team1/ddrive/team1/sam_dataset/internal_dataset/train/image \
#     --train_mask_dir /home/team1/ddrive/team1/sam_dataset/internal_dataset/train/mask \
#     --pmask_dir train_pseudo_masks_internel_ADL_fft_point_30

python3 run_for_train_point_softmax.py \
    --batch_size 1 \
    --patch_size 512 \
    --seed 21 \
    --cam_type adl \
    --model_type vit_b \
    --checkpoint_sam sam_vit_b.pth \
    --checkpoint_cls /home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0717/code/classifier/checkpoints/internel_adl_best.pth \
    --train_image_dir /home/team1/ddrive/team1/sam_dataset/internal_dataset/train/image \
    --train_mask_dir /home/team1/ddrive/team1/sam_dataset/internal_dataset/train/mask \
    --pmask_dir train_point_prompt_pseudo_masks_adl