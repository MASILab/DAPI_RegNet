"""
This script is used to cut the images into pieces of 512x512 pixels.
The steps are as follows:
1. For each tissue, take the round 0,5,10 and n-1
2. Break these into 512x512 pieces, ignore the parts with all black pixels.
3. Save the pictures as numpy arrays in the folder: /home-local/rudravg/Segmentation_test/Images
4. Do the same above process for the DeepCell results and save it in the folder: /home-local/rudravg/Segmentation_test/Masks
"""

from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from my_utils import Utils
import glob
import re

Image.MAX_IMAGE_PIXELS = None

def extract_round_number(filename):
    match = re.search(r'_ROUND_(\d+)_', filename)
    if match:
        return int(match.group(1))
    else:
        return 0
#Step 1: Load the images and apply the masks
tissue_names_file='/nfs2/baos1/rudravg/tissue_names.txt'
original_tissues_path='/nfs2/baos1/rudravg'
DeepCell_path='/nfs2/baos1/rudravg/DeepCell_Results'

raw_dir='/home-local/rudravg/Segmentation_test/Images'
mask_dir='/home-local/rudravg/Segmentation_test/Masks'
#Reading all tissue names
with open(tissue_names_file, 'r') as file:
    tissue_names = [line.strip() for line in file.readlines()]
window_size = (512, 512)
for tissue in tqdm(tissue_names,desc='Generating Images and Truths'):
    search_pattern_raw_files = os.path.join(original_tissues_path, tissue + '*')
    search_pattern_mask_files=os.path.join(DeepCell_path,tissue+'*')
    matching_files_raw = glob.glob(search_pattern_raw_files)
    matching_files_mask = glob.glob(search_pattern_mask_files)
    sorted_files_raw = sorted(matching_files_raw, key=extract_round_number)
    sorted_files_mask = sorted(matching_files_mask, key=extract_round_number)
    files_of_interest_raw = [sorted_files_raw[0], sorted_files_raw[5], sorted_files_raw[10], sorted_files_raw[-1]]
    mask_search_pattern = os.path.join(original_tissues_path, 'Retention_Masks', tissue + '*')
    mask_files = glob.glob(mask_search_pattern)
    if mask_files:
        mask_name = mask_files[0]
        file_of_interest_truth=[sorted_files_mask[0], sorted_files_mask[5], sorted_files_mask[10], sorted_files_mask[-1]]
        mask=np.array(Image.open(mask_name))
        mask = (mask > 0).astype(int)
    else:
        continue
    for i in range(4):
        raw_file=files_of_interest_raw[i]
        truth_file=file_of_interest_truth[i]
        raw_image=np.array(Image.open(raw_file))
        truth_image=np.array(Image.open(truth_file))
        raw_image_masked=(raw_image*mask)/255.
        if(mask.shape!=truth_image.shape):
            print(mask_name,mask.shape)
            print(truth_file,truth_image.shape,i)
        truth_image_masked=(truth_image*mask)/255.
        truth_image_masked=(truth_image_masked>0).astype(int)
        for x in range(0, raw_image_masked.shape[0], window_size[0]):
            for y in range(0, raw_image_masked.shape[1], window_size[1]):
                raw_piece = raw_image_masked[x:x+window_size[0], y:y+window_size[1]]
                truth_piece = truth_image_masked[x:x+window_size[0], y:y+window_size[1]]

                if np.count_nonzero(truth_piece) > 200:
                    raw_path = os.path.join(raw_dir, f'{tissue}{i}_{x}_{y}.npy')
                    truth_path = os.path.join(mask_dir, f'{tissue}{i}_{x}_{y}_mask.npy')
                    np.save(raw_path, (raw_piece))
                    np.save(truth_path, (truth_piece))   