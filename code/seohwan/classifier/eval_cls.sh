#!/bin/bash

echo CLS_EVALUATION >> CLS_EVAL.txt

python3 eval_cls.py \
    --batch_size 64 \
    --patch_size 512 \
    --seed 21 \
    --cls_model_type adl \
    --checkpoint_dir checkpoints/tiger_adl_fft_no_best.pth \
    --test_image_dir /home/team1/ddrive/team1/sam_dataset/TIGER_classifier_train6000by6000/test/image \
    >> CLS_EVAL.txt

    