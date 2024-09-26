#!/bin/bash
DATASET_TYPE=camelyon17 # or camelyon16 

# train patch classifier 
TRAIN_IMAGE_DIR=dataset/$DATASET_TYPE/train/image
VAL_IMAGE_DIR=dataset/$DATASET_TYPE/val/image
TEST_IMAGE_DIR=dataset/$DATASET_TYPE/test/image

python3 1_train_patch_classifier.py \
    --train_image_dir $TRAIN_IMAGE_DIR \
    --val_image_dir $VAL_IMAGE_DIR \
    --test_image_dir $TEST_IMAGE_DIR \ >> logging.txt 

# initial mask generation 
python3 2_initial_mask_generation.py

# sam mask generation 
python3 3_sam_mask_generation.py

