#!/bin/bash

# python3 run_for_train_grad_iter2.py \
#     --batch_size 1 \
#     --patch_size 512 \
#     --seed 21 \
#     --model_type vit_b \
#     --checkpoint_sam sam_vit_b.pth \
#     --checkpoint_decoder /home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0705/code/segmentor/SAM/checkpoints/sam_cam17_iter1_20_gm.pth \
#     --checkpoint_cls /home/team1/ddrive/team1/SAM_WSSS_CAM/gradcam_repo/code/pseudo_mask_generation/ResNet50_vanilla.pth \
#     --train_image_dir /home/team1/ddrive/team1/sam_dataset/camelyon17_classifier_train9000by9000/train/image \
#     --train_mask_dir /home/team1/ddrive/team1/sam_dataset/camelyon17_classifier_train9000by9000/train/mask \
#     --percen80_dir /home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0705/code/pseudo_mask_generation/outputs_vit_b_cam17/train_grad_pseudo_masks_iter1_gm_80

#!/bin/bash

python3 run_for_train_grad.py \
    --batch_size 1 \
    --patch_size 512 \
    --seed 21 \
    --model_type vit_b \
    --checkpoint_sam sam_vit_b.pth \
    --checkpoint_cls /home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0717/code/classifier/checkpoints/cam17_ResNet_best.pth \
    --train_image_dir /home/team1/ddrive/team1/sam_dataset/TIGER_classifier_train6000by6000/val/image \
    --train_mask_dir /home/team1/ddrive/team1/sam_dataset/TIGER_classifier_train6000by6000/val/mask