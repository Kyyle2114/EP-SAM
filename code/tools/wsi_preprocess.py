"""
Code Reference: https://github.com/jpjuvo/camelyon17-multilevel 
"""


# Take a folder where XML files are located and convert each annotation file into a mask (tumorous areas in white, non-tumorous areas in black)
import multiresolutionimageinterface as mir
import cv2
import os
import numpy as np
import pandas as pd 
from tqdm import tqdm 

def CreateAnnotationMask(
    reader,
    annotation_list,
    xml_repository,
    ImageFiles,
    dirAnnotations,
    annotationPath
):    
    # Store only the name of the XML file, excluding the directory and extension, e.g., tumor_001
    fileNamePart = annotationPath.replace('.xml','').replace(dirAnnotations, "")
    
    # Add .tif extension to fileNamePart, e.g., tumor_001.tif
    tifName = fileNamePart + '.tif'

    # If there is no matching value in ImageFiles for tifName, skip -> this is only executed for images with annotations (where tumor exists).
    partialMatches = [s for s in ImageFiles if tifName in s]
    if len(partialMatches) == 0:
        print('Warning - This file is missing from the file list: {0} - skipping.'.format(tifName))
        return
    tifPath = partialMatches[0]
    
    # If the tif file does not exist, skip
    if (not os.path.isfile(tifPath)): 
        print('Warning - Could not locate {0} - skipping this annotation file.'.format(tifPath))
        return
    
    # If a file already exists, skip
    maskPath = tifPath.replace('.tif', '_mask.tif')
    if (os.path.isfile(maskPath)):
        print('Info - Mask file of {0} already exists - skipping'.format(tifPath))
        return
    
    # Fetch the XML file
    xml_repository.setSource(annotationPath)
    xml_repository.load()

    # Convert the XML file with polygons into a mask tif file
    annotation_mask = mir.AnnotationToMask()
    mr_image = reader.open(tifPath)
    if(mr_image is None):
        print('Warning - Could not read {0} - skipping'.format(tifPath))
        return
    label_map = {'metastases': 1, 'normal': 2}
    conversion_order = ['metastases', 'normal']
    annotation_mask.convert(
        annotation_list, 
        maskPath, 
        mr_image.getDimensions(), 
        mr_image.getSpacing(), 
        label_map, 
        conversion_order
    )
    
    return


## This function is adapted from a digital pathology pipeline code of Mikko Tukiainen
# Functions identical to those in the wsi2tissueMask.py file in the utils folder.
# Background is set to 0, and tissue parts to 255
def make_tissue_mask(
    slide, 
    mask_level=4, 
    morpho=None, 
    morpho_kernel_size=5, 
    morpho_iter=1, 
    median_filter=False, 
    return_original=False
):
    ''' make tissue mask
        return tissue mask array which has tissue locations (pixel value 0 -> empty, 255 -> tissue)
    Args:
        slide (MultiResolutionImage): MultiResolutionImage slide to process
        mask_level (int): defines the level of zoom at which the mask will be created (default 4)
        morpho (cv2.MORPHO): OpenCV morpho flag, Cv2.MORPHO_OPEN or Cv2.MORPHO_CLOSE (default None)
        morpho_kernel_size (int): kernel size for morphological transformation (default 5)
        morpho_iter (int): morphological transformation iterations (default=1)
        median_filter (bool): Use median filtering to remove noise (default False)
        return_original (bool): return also the unmasked image
    '''
    # Read the slide
    ds = slide.getLevelDownsample(mask_level)
    original_tissue = slide.getUCharPatch(0,
                                          0,
                                          int(slide.getDimensions()[0] / float(ds)),
                                          int(slide.getDimensions()[1] / float(ds)),
                                          mask_level)
    
    # Extract only the brightness channel of the mask and binarize according to the threshold
    tissue_mask = cv2.cvtColor(np.array(original_tissue), cv2.COLOR_RGBA2RGB)
    tissue_mask = cv2.cvtColor(tissue_mask, cv2.COLOR_BGR2HSV)
    tissue_mask = tissue_mask[:, :, 1]
    _, tissue_mask = cv2.threshold(tissue_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply morphological transformations
    if morpho is not None:
        kernel = np.ones((morpho_kernel_size, morpho_kernel_size), np.uint8)
        tissue_mask = cv2.morphologyEx(tissue_mask, morpho, kernel, iterations=morpho_iter)
    
    # Remove noise with median filtering
    if median_filter:
        tissue_mask = cv2.medianBlur(tissue_mask, 15)
    
    # Convert mask to numpy array
    tissue_mask = np.array(tissue_mask, dtype=np.uint8)

    # Decide whether to also return the original
    if return_original:
        return tissue_mask, original_tissue
    else:
        return tissue_mask


def CreateTissueMask(
    dirData,
    reader,
    tifPath
):  
    # Extract only the filename
    fileNamePart = tifPath.replace('.tif','').replace(dirData, "")
    
    # Skip if this mask is already found
    maskPath = tifPath.replace('.tif', '_tissue_mask_ds16.npy')
    if (os.path.isfile(maskPath)):
        print('Info - Tissue mask file of {0} already exists - skipping'.format(tifPath))
        return
    
    # Create tissue mask
    mr_image = reader.open(tifPath)
    if(mr_image is None):
        print('Warning - Could not read {0} - skipping'.format(tifPath))
        return
    tissue_mask = make_tissue_mask(mr_image,
                                   # mr_image.getBestLevelForDownSample(16), 
                                   1,
                                   morpho=cv2.MORPH_CLOSE,
                                   morpho_kernel_size=7,
                                   morpho_iter=2,
                                   median_filter=True)
    # tissue_mask is a binary array dtype.uint8 (16 times downsampled)
    np.save(maskPath, tissue_mask)
    
    return


def getTissueMask(tifPath):
    maskPath = tifPath.replace('.tif', '_tissue_mask_ds16.npy')
    if (not os.path.isfile(maskPath)): return None
    return np.load(maskPath)


# Function to create the center positions of samples (patches)
def sample_centers(
    tissue_mask, 
    mask_downscale=16, 
    sample_side=512, 
    focus_width_percentage=0.25, 
    padding_percentage=0.01
):
    # Width and height of the tissue mask
    mask_width, mask_height = tissue_mask.shape[:2]

    # Sample size
    side = sample_side / mask_downscale

    # Padding size
    padding_width = mask_width * padding_percentage
    padding_height = mask_height * padding_percentage

    # Half-width of the focus area
    half_focus = int(sample_side * focus_width_percentage / mask_downscale)
    
    # List to store the center coordinates of the samples
    sample_centers = []
    
    # Determine sample centers based on areas where tissue exists
    for i in range(int(mask_width // side)):
        for j in range(int(mask_height // side)):
            for sub_shift in [0, 0.5]:
                x = int((i + sub_shift) * side)
                y = int((j + sub_shift) * side)
                min_x = int(max(0, x - half_focus))
                max_x = int(min(x + half_focus, mask_width - 1))
                min_y = int(max(0, y - half_focus))
                max_y = int(min(y + half_focus, mask_height - 1))
                
                # Skip samples in the padding area
                if(min_x < padding_width or max_x > mask_width - padding_width): continue
                if(min_y < padding_height or max_y > mask_height - padding_height): continue
                
                # Add to samples only areas where tissue exists
                if(tissue_mask[min_x:max_x, min_y:max_y].sum() > 0):
                    sample_centers.append(np.array([x, y]))
                    
    # Restore the mask downscale to compute coordinates
    sample_centers = np.array(sample_centers) * mask_downscale
    return sample_centers


# Check if there is a tumor in the patch
def isTumor(mask_level_0):
    return (mask_level_0.max() > 0)


# Calculate the percentage of tumor in the patch
def tumorPercentage(mask_level_0):
    area = mask_level_0.shape[0] * mask_level_0.shape[1]
    tumorPixels = np.count_nonzero(mask_level_0)
    channels = 3
    return tumorPixels / (area * channels)


# Load image
def getImage(reader, tifPath):
    if (not os.path.isfile(tifPath)): return None
    return reader.open(tifPath)


# Load mask file (only the tumorous parts)
def getAnnoMask(reader, tifPath):
    maskPath = tifPath.replace('.tif', '_mask.tif')
    if (not os.path.isfile(maskPath)): return None
    return reader.open(maskPath)


def getSamplesWithAnnotations(mr_image, mr_mask, x_cent, y_cent, width=512, height=512):
    channels = 3
    imgs = np.zeros((1, width, height, channels), dtype=np.int32)
    masks = np.zeros((1, width, height, channels), dtype=np.int32)

    lev = mr_image.getBestLevelForDownSample(1)
    ds = mr_image.getLevelDownsample(lev)
    imgs[0] = mr_image.getUCharPatch(int(x_cent - (ds*width/2)),
                                     int(y_cent - (ds*height/2)),
                                     width,
                                     height,
                                     lev)
    masks[0] = mr_mask.getUCharPatch(int(x_cent - (ds*width/2)),
                                     int(y_cent - (ds*height/2)),
                                     width,
                                     height,
                                     lev)
    return imgs, masks


def getSamples(mr_image, x_cent, y_cent, levels, sz):
    channels = 3
    imgs = np.zeros((len(levels), sz, sz, channels), dtype=np.uint8)
    for i, lev in enumerate(levels):
        ds = mr_image.getLevelDownsample(lev)
        imgs[i] = mr_image.getUCharPatch(int(x_cent - (ds*sz/2)),
                                         int(y_cent - (ds*sz/2)),
                                         sz,
                                         sz,
                                         lev)
    return imgs


def getMaskedSamples(mr_mask, x_cent, y_cent, levels, sz):
    masks = np.zeros((len(levels), sz, sz), dtype=np.uint8)
    for i, lev in enumerate(levels):
        ds = mr_mask.getLevelDownsample(lev)
        mask = mr_mask.getUCharPatch(int(x_cent - (ds*sz/2)),
                                     int(y_cent - (ds*sz/2)),
                                     sz,
                                     sz,
                                     lev)
        # Select and use only the red channel.
        masks[i] = mask[:, :, 0]  # Using only the red channel
    return masks


# Split a WSI file into patches and create a CSV file storing annotations for each patch
def CreateDF_Camelyon16(dirData, dirHome, tifPath, overrideExisting=False, size=256):
    # How many times to multiply the 16x reduced tissue mask. Since it's 16, it's equivalent to 400x original magnification
    mask_downscales = [16]
    mags = ['400x']
    
    mr_image = getImage(tifPath)
    mr_mask = getAnnoMask(tifPath)

    for i, (mask_downscale, mag) in enumerate(zip(mask_downscales, mags)):
        
        # Only store the file name
        fileNamePart = tifPath.replace('.tif','').replace(dirData, "")
        df_path = dirHome + '/dataframes/' + fileNamePart.split('/')[-1] + '.csv'
    
        if (os.path.isfile(df_path) and not overrideExisting):
            print('Info - Dataframe file of {0} already exists - skipping'.format(df_path))
            continue
        
        tissue_mask = getTissueMask(tifPath)
        patch_centers = sample_centers(tissue_mask, mask_downscale=mask_downscale, sample_side=size)

        print("Sliced WSI {1} to {0} patches.".format(len(patch_centers), tifPath))
        
        # Load the current image file/mask file
        df = pd.DataFrame(columns=[
            'patchId',
            'fileName',
            'centerX',
            'centerY',
            'isTumor',
            'tumorPercentage'
        ])
        
        # If the directory is different, it needs to be changed.
        wsi_name = tifPath.split('/')[-1]
        tumor_idx = wsi_name.strip('.tif').split('_')[-1]
        
        for c in tqdm(patch_centers, 'Patches...'):
            imgs, masks = getSamplesWithAnnotations(mr_image, mr_mask, x_cent=c[1], y_cent=c[0], width=size, height=size)

            isTumor_attr = isTumor(masks[i])
            tumorPrc_attr = tumorPercentage(masks[i])
            
            df = df.append({
                'patchId': str(tumor_idx) + '_' + str(c[0]).zfill(7) + str(c[1]).zfill(7),
                'fileName': tifPath,
                'centerX': c[0],
                'centerY': c[1],
                'isTumor': isTumor_attr,
                'tumorPercentage': int(tumorPrc_attr * 1000) / 10
            }, ignore_index=True)
    
        df.to_csv(df_path)
        
        return 


# Split a WSI file into patches and create a CSV file storing annotations for each patch
def CreateDF_Camelyon17(dirData, dirHome, tifPath, overrideExisting=False):
    
    # Only store the file name
    fileNamePart = tifPath.replace('.tif','').replace(dirData, "")
    df_path = dirHome + '/dataframes/' + fileNamePart.split('/')[-1] + '.csv'

    if (os.path.isfile(df_path) and overrideExisting == False):
        print('Info - Dataframe file of {0} already exists - skipping'.format(tifPath))
        return
    
    tissue_mask = getTissueMask(tifPath)
    patch_centers = sample_centers(tissue_mask)

    print("Sliced WSI {1} to {0} patches.".format(len(patch_centers), tifPath))
    
    # Load current image file / mask file
    mr_image = getImage(tifPath)
    mr_mask = getAnnoMask(tifPath)
    
    df = pd.DataFrame(columns=[
        'patchId',
        'fileName',
        'center',
        'patient',
        'node',
        'centerX',
        'centerY',
        'isTumor',
        'tumorPercentage'
    ])
    
    # If the directory is different, it needs to be changed.
    split = tifPath.split('/')
    cnt = int(split[-3].strip('center_'))
    splitpatient = split[-1].split('_')
    patient = int(splitpatient[1])
    node = int(splitpatient[3].strip('.tif'))
    
    for c in tqdm(patch_centers, 'Patches...'):
        img, mask = getSamplesWithAnnotations(mr_image, mr_mask, x_cent=c[1], y_cent=c[0], width=512, height=512)
        isTumor_attr = isTumor(mask[0])
        tumorPrc_attr = tumorPercentage(mask[0])
        
        df = df.append({
            'patchId': str(patient) + str(node).zfill(3) + '_' + str(patient) + str(0) + str(c[0]).zfill(7) + str(c[1]).zfill(7),
            'fileName': tifPath,
            'center': cnt,
            'patient': patient,
            'node': node,
            'centerX':c[0],
            'centerY':c[1],
            'isTumor':isTumor_attr,
            'tumorPercentage': int(tumorPrc_attr * 1000)/10
        }, ignore_index=True)

    df.to_csv(df_path)
    
    return 


def ExtractPatches(
    ImageFiles,
    reader,
    df_list,
    dirName_list,
    dirRoot,
    levels,
    dataset_type
):
    for WSI in tqdm(ImageFiles):
        mr_image = reader.open(WSI)
        mask_image_path = WSI.replace('.tif', '_mask.tif')
        mask_image = reader.open(mask_image_path)
        split = WSI.split('/')
        
        if dataset_type == 'camelyon16':
            wsi_id = split[-1].split('_')[1].split('.')[0]
        
        elif dataset_type == 'camelyon17':
            splitpatient = split[-1].split('_')
            patient = int(splitpatient[1])
            node = int(splitpatient[3].strip('.tif'))
        
        for df, dirName in zip(df_list, dirName_list):
            
            if dataset_type == 'camelyon16':
                df_sub = df[df.wsi_id == wsi_id]
            
            elif dataset_type == 'camelyon17':
                df_sub = df[(df.patient == patient) & (df.node == node)]

            for i in range(len(df_sub)):
                id = str(df_sub.iloc[i].patchId)
                label = 1 if df_sub.iloc[i].isTumor else 0

                image_dir = os.path.join(dirRoot, dirName, 'image')
                mask_dir = os.path.join(dirRoot, dirName, 'mask')
                os.makedirs(image_dir, exist_ok=True)  
                os.makedirs(mask_dir, exist_ok=True)   
                
                fileNamePrefix = os.path.join(image_dir, id)
                center_x = df_sub.iloc[i].centerX
                center_y = df_sub.iloc[i].centerY
        
                imgs = getSamples(mr_image, center_y, center_x, levels, 512)
                image_file_name = f"{fileNamePrefix}_{label}.png"
                
                if not os.path.exists(image_file_name):
                    cv2.imwrite(image_file_name, imgs[0])
                    
                else:
                    print(f"File {image_file_name} already exists, skipping.")
        
                if label:
                    masks = getMaskedSamples(mask_image, center_y, center_x, levels, 512)
                    maskFileNamePrefix = os.path.join(mask_dir, id)
                    mask_file_name = f"{maskFileNamePrefix}_{label}.png"
                    
                    if not os.path.exists(mask_file_name):
                        cv2.imwrite(mask_file_name, masks[0])
                        
                    else:
                        print(f"File {mask_file_name} already exists, skipping.")
                        
    return 