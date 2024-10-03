#!/bin/bash

TEST_DATASET_DIR=dataset/camelyon17/test
SAM_MODEL_TYPE='vit_b'
SAM_CHECKPOINT='sam_vit_b.pth'
RESNET_CHECKPOINT='checkpoints/camelyon17_resnet_adl.pth'
SAM_DECODER_CHECKPOINT='checkpoints/camelyon17_decoder_iter3.pth'

python3 model_inference.py \
    --sam_model_type $SAM_MODEL_TYPE \
    --sam_checkpoint $SAM_CHECKPOINT \
    --resnet_checkpoint $RESNET_CHECKPOINT \
    --sam_decoder_checkpoint $SAM_DECODER_CHECKPOINT \
    --test_dataset_dir $TEST_DATASET_DIR \
    >> logging_inference.txt