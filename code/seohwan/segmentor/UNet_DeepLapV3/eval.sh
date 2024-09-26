#!/bin/bash

# TEST_IMAGE_DIR=dataset/test/image
# TEST_MASK_DIR=dataset/test/mask

echo DEEPLAB_EVAL >> deeplab_eval_cam17.txt
python3 eval.py \
    --batch_size 1 \
    --seed 21 \
    --model deeplab \
    --checkpoint_dir /home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0705/code/segmentor/UNet_DeepLapV3/Jul19_160058_deeplab_cam17.pth \
    --test_image_dir /home/team1/ddrive/team1/sam_dataset/camelyon17_classifier_train6000by6000/test/image \
    --test_mask_dir /home/team1/ddrive/team1/sam_dataset/camelyon17_classifier_train6000by6000/test/mask \
    >> deeplab_eval_cam17.txt 

# echo DEEPLAB_EVAL >> deeplab_eval_tiger.txt 
# python3 eval.py \
#     --batch_size 1 \
#     --seed 21 \
#     --model deeplab \
#     --checkpoint_dir /home/team1/ddrive/team1/SAM_WSSS_CAM/SAM_WSSS_CAM_0705/code/segmentor/UNet_DeepLapV3/Jul19_160116_deeplab_TIGER.pth \
#     --test_image_dir /home/team1/ddrive/team1/sam_dataset/TIGER_classifier_train6000by6000/test/image \
#     --test_mask_dir /home/team1/ddrive/team1/sam_dataset/TIGER_classifier_train6000by6000/test/mask \
#     >> deeplab_eval_tiger.txt
 