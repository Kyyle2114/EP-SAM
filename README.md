# Weakly supervised pseudo-labeling for breast cancer segmentation based on SAM: Leveraging Prompts for Effective Knowledge Transfer

Joonhyeon Song, Seohwan Yun, Seongho Yoon, Joohyeok Kim, Sangmin Lee

## Abstract 

> Herein, we propose a novel approach beyond supervised learning for effective pathological image analysis, addressing the challenge of limited robust labeled data. Pathological diagnosis of diseases like cancer has conventionally relied on the evaluation of morphological features by physicians and pathologists. However, recent advancements in compute-aided diagnosis (CAD) systems are gaining significant attention as diagnostic support tools. Although the advancement of deep learning has improved CAD significantly, segmentation models typically require large pixel-level annotated dataset, and such labeling is expensive. Existing studies not based on supervised approaches still struggle with limited generalization , and no practical approach has emerged yet.  To address this issue, we present a a weakly supervised semantic segmentation (WSSS) model by combining multiple instance learning and class activation map-based pseudo-labeling. For effective pretraining, we adopt the segment anything model (SAM)—a foundation model that is pretrained on large datasets and operates in zero-shot configurations using only coarse prompts. The proposed approach obtains enhanced knowledge of the attention dropout layer via the explicit visual prompting of SAM, which generates pseudo-labels.  To address the superiority of the proposed method, experimental studies are conducted on histopathological breast cancer datasets. The proposed method outperformed other WSSS methods across three open datasets, demonstrating its efficiency by achieving this with only 12GB of GPU memory during training.

## Overview
 
![](./asset/figure1.png)

We devised a framework that transfers the enhanced knowledge of ADL through explicit visual prompting to SAM, generating pseudo-labels.

## Environment

The following describes our experimental setup.

- Ubuntu 22.04.4 LTS
- 8x Intel(R) Xeon(R) Silver 4112 CPU @ 2.60GHz
- 2x NVIDIA TITIAN Xp , 12GB

Execute the following cell to set up a virtual environment for running the code.

```bash 
# create a virtual env and activate
conda create -n wsplf python=3.10
conda activate wsplf 

# download packages
pip install -r requirements.txt 
```

Additional packages such as ASAP are required to extract patches from WSI. Install the following two packages to run the code:

- [OpenSlide](https://openslide.org/download/) 3.4.1 (ASAP 1.8 depends on libopenslide)
- [ASAP](https://github.com/computationalpathologygroup/ASAP) 1.8 (1.9 does not support the Philips scanner TIFF file format of the Center_4)

## How to Use 

### Train

- Move to ```code/``` directory

- Move downloaded datasets to ```dataset/camelyon16``` or ```dataset/camelyon17``` according to their respective dataset types

- Modify the ```DATASET_TYPE``` in ```train.sh``` file to match your downloaded dataset

- Run ```./train.sh```. This will perform everything from patch extraction to model training. Training logs will be saved in ```logging.txt```

- Model weights will be saved in ```code/checkpoints```

### Inference 

> TBD

## Dataset

We utilized two open public breast cancer WSI datasets: Camelyon16 & Camelyon17. 

### Camelyon16

- [Dataset link](https://camelyon16.grand-challenge.org/)

### Camelyon17

- [Dataset link](https://camelyon17.grand-challenge.org/)
