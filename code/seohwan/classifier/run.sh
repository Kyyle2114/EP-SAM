# #!/bin/bash

python3 run_cls.py \
    --batch_size 16 \
    --patch_size 512 \
    --seed 22 \
    --epoch 50 \
    --lr 1e-5 \
    --weight_decay 1e-3 \
    --project_name cam16_adl_fft \
    --cls_model_type adl_fft \
    --cuda 1 \
    --train_image_dir /home/team1/ddrive/team1/sam_dataset/camelyon16_classifier_train6000by6000/train/image \
    --val_image_dir /home/team1/ddrive/team1/sam_dataset/camelyon16_classifier_train6000by6000/val/image \
    --checkpoint_dir cam16_adl_fft_1e-5 \
