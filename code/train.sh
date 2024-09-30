#!/bin/bash

# download checkpoints for Segment Anything (ViT-B)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
mv sam_vit_b_01ec64.pth sam_vit_b.pth

# WSI preprocessing
DATASET_TYPE=camelyon17 # or camelyon16 

python3 0_data_preprocess.py \
    --dataset_type $DATASET_TYPE \

# train patch classifier 
TRAIN_DATASET_DIR=dataset/$DATASET_TYPE/train
VAL_DATASET_DIR=dataset/$DATASET_TYPE/val
TEST_DATASET_DIR=dataset/$DATASET_TYPE/test

python3 1_train_patch_classifier.py \
    --train_image_dir $TRAIN_DATASET_DIR/image \
    --val_image_dir $VAL_DATASET_DIR/image \
    --test_image_dir $TEST_DATASET_DIR/image \
    >> logging.txt

# initial mask generation 
python3 2_generate_initial_mask.py \
    --dataset_type $DATASET_TYPE \
    --train_image_dir $TRAIN_DATASET_DIR/image \
    >> logging.txt

# preliminary mask decoder fine-tuning
SAM_MODEL_TYPE='vit_b'
SAM_CHECKPOINT='sam_vit_b.pth'

python3 3_preliminary_fine_tuning.py \
    --sam_model_type $SAM_MODEL_TYPE \
    --sam_checkpoint $SAM_CHECKPOINT \
    --train_dataset_dir $TRAIN_DATASET_DIR \
    --val_dataset_dir $VAL_DATASET_DIR \
    >> logging.txt

# iterative re-training
python3 4_iterative_re_training.py \
    --n_iter 3

