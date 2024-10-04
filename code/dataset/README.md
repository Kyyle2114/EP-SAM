# Download dataset 

You can download the datasets using [AWS CLI](https://aws.amazon.com/cli/?nc1=h_ls) commands below. For alternative download methods(e.g., Baidu), please refer to the respective dataset links.

```bash
cd code

# camelyon16
aws s3 cp --no-sign-request s3://camelyon-dataset/CAMELYON16/images/ dataset/camelyon16/images/ --recursive
aws s3 cp --no-sign-request s3://camelyon-dataset/CAMELYON16/annotations/ dataset/camelyon16/annotations/ --recursive

# camelyon17
aws s3 cp --no-sign-request s3://camelyon-dataset/CAMELYON17/images/ dataset/camelyon17/images/ --recursive
aws s3 cp --no-sign-request s3://camelyon-dataset/CAMELYON17/annotations/ dataset/camelyon17/annotations/ --recursive
```

# Dataset structure (WSI preprocessing)

For WSI preprocessing, the dataset structure should be organized as follows:

```
├── LICENSE
├── README.md
├── asset
├── code
│   ├── dataset
│   │   ├── README.md
│   │   ├── camelyon16
│   │   │   ├── annotations
│   │   │   │   └── ...
│   │   │   └── images
│   │   │       └── ...
│   │   └── camelyon17
│   │       ├── annotations
│   │       │   └── ...
│   │       └── images
│   │           └── ...
│   ├── ...
│   
└── requirements.txt
```

# Dataset structure (Model Training / Inference)

After WSI preprocessing, the dataset structure for model training will be as follows:

```
├── LICENSE
├── README.md
├── asset
├── code
│   ├── dataset
│   │   ├── README.md
│   │   ├── camelyon16
│   │   │   ├── annotations
│   │   │   ├── images
│   │   │   ├── test
│   │   │   │   ├── image
│   │   │   │   │   └── ...
│   │   │   │   └── mask
│   │   │   │       └── ...
│   │   │   ├── train
│   │   │   │   ├── image
│   │   │   │   │   └── ...
│   │   │   │   └── mask
│   │   │   │       └── ...
│   │   │   └── val
│   │   │       ├── image
│   │   │       │   └── ...
│   │   │       └── mask
│   │   │           └── ...
│   │   └── camelyon17
│   │       ├── annotations
│   │       ├── images
│   │       ├── test
│   │       │   ├── image
│   │       │   │   └── ...
│   │       │   └── mask
│   │       │       └── ...
│   │       ├── train
│   │       │   ├── image
│   │       │   │   └── ...
│   │       │   └── mask
│   │       │       └── ...
│   │       └── val
│   │           ├── image
│   │           │   └── ...
│   │           └── mask
│   │               └── ...
│   ├── ...
│   
└── requirements.txt
```