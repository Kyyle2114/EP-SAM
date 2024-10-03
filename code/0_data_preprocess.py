import warnings
warnings.filterwarnings('ignore')

# Download the executable file from https://github.com/computationalpathologygroup/ASAP/releases, and append the binary folder to the path.
import sys
sys.path.append('/opt/ASAP/bin')

# Import the multiresolutionimageinterface.py file from the bin folder.
import multiresolutionimageinterface as mir
import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from tools.wsi_preprocess import *

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset_type', type=str, default='camelyon17', choices=['camelyon16', 'camelyon17'], help='dataset type')
    
    return parser

def main(opts):
    """
    WSI Proprocessing

    Args:
        opts (argparser): argparser
    """
    ### Convert lesion annotations to masks ###
    
    reader = mir.MultiResolutionImageReader()
    annotation_list = mir.AnnotationList()
    xml_repository = mir.XmlRepository(annotation_list)
    
    if opts.dataset_type == 'camelyon16':
        dirAnnotations = f'dataset/{opts.dataset_type}/annotations'
        dirData = f'dataset/{opts.dataset_type}/images'
        dirHome = f'dataset/{opts.dataset_type}'
        
    elif opts.dataset_type == 'camelyon17':
        dirAnnotations = f'dataset/{opts.dataset_type}/annotations'
        dirData = f'dataset/{opts.dataset_type}/images'
        dirHome = f'dataset/{opts.dataset_type}'
    
    ImageFiles = []
    AnnotationFiles = []
    
    for r, d, f in os.walk(dirData):
        for file in f:
            if '.tif' in file and 'mask' not in file:
                ImageFiles.append(os.path.join(r, file))
                    
    for r, d, f in os.walk(dirAnnotations):
        for file in f:
            if '.xml' in file:
                AnnotationFiles.append(os.path.join(r, file))

    ImageFiles.sort()
    AnnotationFiles.sort()
    
    print('Create Annotation Mask')
    for file_path in tqdm(AnnotationFiles):
        CreateAnnotationMask(
            reader=reader,
            annotation_list=annotation_list,
            xml_repository=xml_repository,
            ImageFiles=ImageFiles,
            dirAnnotations=dirAnnotations,
            annotationPath=file_path
        )
    
    ### Making tissue masks ###
    
    print('Create Tissue Mask')
    for file_path in tqdm(ImageFiles):
        CreateTissueMask(
            dirData=dirData,
            reader=reader,
            tifPath=file_path
        )
    
    ### Create a DataFrames to record patch information ###
    
    print('Create Dataframes')
    for file_path in tqdm(ImageFiles):
        try:
            if opts.dataset_type == 'camelyon16':
                CreateDF_Camelyon16(
                    dirData=dirData,
                    dirHome=dirHome,
                    tifPath=file_path
                )
                
            elif opts.dataset_type == 'camelyon17':
                CreateDF_Camelyon17(
                    dirData=dirData,
                    dirHome=dirHome,
                    tifPath=file_path
                )
        except:
            print(f'Cannot find {file_path} file')
            
    ### Creating dataset ###
            
    directory = dirHome + '/dataframes'
    dfs = []
    
    if opts.dataset_type == 'camelyon16':
        for filename in os.listdir(directory):
            if filename.endswith('.csv'):
                filepath = os.path.join(directory, filename)
                df = pd.read_csv(filepath)
                dfs.append(df)

        merged_df = pd.concat(dfs, axis=0, ignore_index=True)
        merged_df = merged_df.drop('Unnamed: 0', axis=1)
        merged_df['wsi_id'] = merged_df['patchId'].apply(lambda x: x.split('_')[0])
        merged_df = merged_df[(merged_df['tumorPercentage'] > 20) & (merged_df['tumorPercentage'] < 90)]
        
        train_comb = [
            '061', '018', '039', '031', '035', '051', '064', '040', '030', '007', 
            '025', '041', '054', '065', '005', '046', '023', '004', '022', '008', 
            '006', '068', '001', '009', '015', '042', '016', '056', '027', '057', 
            '038', '063', '019', '045', '037', '062', '058', '033', '010', '002', 
            '060', '055', '069', '052', '066', '011', '024'
        ]

        val_comb = ['048', '013', '067', '044', '043', '036']

        test_comb = [
            '032', '047', '049', '028', '050', '053', '034' ,'012' ,'059', '014', 
            '020' ,'003', '026','029' ,'017'
        ]
    
    elif opts.dataset_type == 'camelyon17':
        for filename in os.listdir(directory):
            if filename.endswith('.csv'):
                parts = filename.split('_')
                if len(parts) > 1 and not ('080' <= parts[1] <= '099'):
                    filepath = os.path.join(directory, filename)
                    df = pd.read_csv(filepath)
                    dfs.append(df)

        merged_df = pd.concat(dfs, axis=0, ignore_index=True)
        merged_df = merged_df.drop('Unnamed: 0', axis=1)
        merged_df['wsi_id'] = merged_df['patchId'].apply(lambda x: x.split('_')[0])
        merged_df = merged_df[(merged_df['tumorPercentage'] > 20) & (merged_df['tumorPercentage'] < 90)]
        
        train_comb = [
            '241', '512', '204', '224', '444', '481', '423', '603', '674',
            '213', '161', '363', '451', '151', '174', '720', '202', '410',
            '171', '640', '382', '120', '662', '681', '402', '464', '731'
        ]

        val_comb = ['343', '521', '152']

        test_comb = ['622', '391', '044', '614', '104', '091', '754', '463', '172']
    
    df_train = merged_df[merged_df['wsi_id'].isin(train_comb)]
    df_valid = merged_df[merged_df['wsi_id'].isin(val_comb)]
    df_test = merged_df[merged_df['wsi_id'].isin(test_comb)]
    
    # Training - positive set 
    df_train_copy = df_train.copy()
    bins = np.arange(0, 105, 5)
    df_train_copy['range'] = pd.cut(df_train_copy['tumorPercentage'], bins, right=False)
    
    def sample_per_group(x):
        # 데이터셋마다 숫자가 다른 이유는? - 이 부분 물어볼 것 
        n_samples = 430 if opts.dataset_type == 'camelyon16' else 500
        return x.sample(n=min(len(x), n_samples), random_state=42) if len(x) > 0 else x

    df_train_sampled = df_train_copy.groupby('range', as_index=False, observed=True).apply(sample_per_group).reset_index(drop=True)
    df_train_sampled.drop('range', axis=1, inplace=True)
    df_train_sampled.to_csv(dirHome + '/sample_patches_train.csv', index=False)
    
    # Training - negative set 
    df_train_neg_all = merged_df[(merged_df['wsi_id'].isin(train_comb)) & (merged_df['isTumor'] == False)]
    df_train_neg = df_train_neg_all.sample(n=6000, random_state=42)
    df_train_neg.to_csv(dirHome + '/sample_patches_negative_train.csv', index=False)
    
    # Valid - positive set
    df_valid_copy = df_valid.copy()
    bins = np.arange(0, 105, 5)
    df_valid_copy['range'] = pd.cut(df_valid_copy['tumorPercentage'], bins, right=False)

    def sample_per_group(x):
        n_samples = 50 # 두 데이터 모두 50으로 동일?
        return x.sample(n=min(len(x), n_samples), random_state=42) if len(x) > 0 else x

    df_valid_sampled = df_valid_copy.groupby('range', as_index=False, observed=True).apply(sample_per_group).reset_index(drop=True)
    
    # 왜 17만 if block 실행하는지 물어볼 것 
    if opts.dataset_type == 'camelyon17':
        shortfall = 1000 - len(df_valid_sampled)
        if shortfall > 0:
            additional_samples = df_valid.loc[~df_valid.index.isin(df_valid_sampled.index)].sample(n=shortfall, random_state=42)
            df_valid_sampled = pd.concat([df_valid_sampled, additional_samples])
    
    df_valid_sampled.drop('range', axis=1, inplace=True)
    df_valid_sampled.to_csv(dirHome + '/sample_patches_valid.csv', index=False)
    
    # Valid - negative set
    df_valid_neg_all = merged_df[(merged_df['wsi_id'].isin(val_comb)) & (merged_df['isTumor'] == False)]
    df_valid_neg = df_valid_neg_all.sample(n=700, random_state=42)
    df_valid_neg.to_csv(dirHome + '/sample_patches_negative_valid.csv', index=False)   
    
    # Test - positive set
    # 16 코드를 기준으로 작성 .. 17에서도 문제 없는지 확인 필요 
    df_test_copy = df_test.copy()
    bins = np.arange(0, 105, 5)
    df_test_copy['range'] = pd.cut(df_test_copy['tumorPercentage'], bins, right=False)

    def sample_per_group(x):
        n_samples = 143 if opts.dataset_type == 'camelyon16' else 100 # 16 코드 참고해서 구현, 17의 경우 n_samples 몇으로 해야 하는지?  
        return x.sample(n=min(len(x), n_samples), random_state=42) if len(x) > 0 else x

    df_test_sampled = df_test_copy.groupby('range', as_index=False, observed=True).apply(sample_per_group).reset_index(drop=True)
    df_test_sampled.drop('range', axis=1, inplace=True)
    df_test_sampled.to_csv(dirHome + '/sample_patches_test.csv', index=False)

    # Test - negative set
    df_test_neg_all = merged_df[(merged_df['wsi_id'].isin(test_comb)) & (merged_df['isTumor'] == False)]
    df_test_neg = df_test_neg_all.sample(n=2000, random_state=42)
    df_test_neg.to_csv(dirHome + '/sample_patches_negative_test.csv', index=False)
    
    ### Creating Patches ### 
    
    df_train = pd.read_csv(dirHome + '/sample_patches_train.csv')
    df_valid = pd.read_csv(dirHome + '/sample_patches_valid.csv')
    df_test = pd.read_csv(dirHome + '/sample_patches_test.csv')
    
    # 16 코드의 df_train['wsi_id'] = df_train['patchId'].apply(lambda x: x.split('_')[0]) 부분은 제거했는데, 왜 필요한지 ? 
    
    # Extract positive patches
    dirRoot = f'dataset/{opts.dataset_type}'
    df_list = [df_train, df_valid, df_test]
    dirName_list = ['train', 'val', 'test']
    levels = [0]

    ExtractPatches(
        ImageFiles=ImageFiles,
        reader=reader,
        df_list=df_list,
        dirName_list=dirName_list,
        dirRoot=dirRoot,
        levels=levels,
        dataset_type=opts.dataset_type
    )

    # Extract negative patches     
    df_train_neg = pd.read_csv(dirHome + '/sample_patches_negative_train.csv')
    df_valid_neg = pd.read_csv(dirHome + '/sample_patches_negative_valid.csv')
    df_test_neg = pd.read_csv(dirHome + '/sample_patches_negative_test.csv')
    
    # 마찬가지로 16 코드의 df_train['patchId'].apply(lambda x: x.split('_')[0]) 부분은 제거했는데, 왜 필요한지 ? 
    df_list = [df_train_neg, df_valid_neg, df_test_neg]
    dirName_list = ['train', 'val', 'test']
    levels = [0]
    
    ExtractPatches(
            ImageFiles=ImageFiles,
            reader=reader,
            df_list=df_list,
            dirName_list=dirName_list,
            dirRoot=dirRoot,
            levels=levels,
            dataset_type=opts.dataset_type
        )

    return
    
if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser('WSI Preprocessing', parents=[get_args_parser()])
    opts = parser.parse_args()
    
    print('WSI Preprocessing')
    
    main(opts)
    
    print('=== DONE === \n')    