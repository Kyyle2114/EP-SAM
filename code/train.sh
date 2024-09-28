#!/bin/bash
DATASET_TYPE=camelyon17 # or camelyon16 

# train patch classifier 
TRAIN_IMAGE_DIR=dataset/$DATASET_TYPE/train/image
VAL_IMAGE_DIR=dataset/$DATASET_TYPE/val/image
TEST_IMAGE_DIR=dataset/$DATASET_TYPE/test/image

python3 1_train_patch_classifier.py \
    --train_image_dir $TRAIN_IMAGE_DIR \
    --val_image_dir $VAL_IMAGE_DIR \
    --test_image_dir $TEST_IMAGE_DIR \
    >> logging.txt

# initial mask generation 
python3 2_generate_initial_mask.py \
    --dataset_type $DATASET_TYPE \
    --train_image_dir $train_image_dir \
    >> logging.txt

# preliminary mask decoder fine-tuning
python3 3_preliminary_fine_tuning.py


# iterative re-training
python3 4_iterative_re_training.py

