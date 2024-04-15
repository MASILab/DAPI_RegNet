from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from my_utils import Utils
import glob
import re
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm
from my_utils import Utils
import torch
import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

raw_dir='/home-local/rudravg/Segmentation_test/Images'
mask_dir='/home-local/rudravg/Segmentation_test/Masks'
model_path='/home-local/rudravg/test_DAPI/epochs/epoch_35/epoch_35.pth'

#Reading all tissue names
with open(tissue_names_file, 'r') as file:
    tissue_names = [line.strip() for line in file.readlines()]
model,device=Utils.load_model(model_path)
device='cuda:0'
model.to(device)
model.eval()
with open('/nfs2/forGaurav/yesbackup/DAPI_RegSegNet/final_metricsoutput_version2_unmasked.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['tissue', 'round_x', 'before_ncc', 'after_ncc','intensity','mean_rmse'])
    for tissue in tqdm(tissue_names,desc='Generating Images and Truths'):
        search_pattern_raw_files = os.path.join(original_tissues_path, tissue + '*')
        matching_files_raw = glob.glob(search_pattern_raw_files)
        sorted_files_raw = sorted(matching_files_raw, key=extract_round_number) #This is important
        mask_search_pattern = os.path.join(original_tissues_path, 'Retention_Masks', tissue + '*')
        mask_files = glob.glob(mask_search_pattern)
        round_0=sorted_files_raw[0]
        if mask_files:
            mask_name = mask_files[0]
            mask=np.array(Image.open(mask_name))
            mask = (mask > 0).astype(int)
        else:
            print(f"Bruh fix mask for {tissue}")
            continue
        for round_x in tqdm(sorted_files_raw[1:]):
            moving,fixed,orig_height,orig_width=Utils.load_images_and_no_mask(fixed=round_0,moving=round_x,mask=mask_name)
            registered_tissue,L2_warp=Utils.register_tissues(moving=moving,fixed=fixed,model=model,device=device)
            moving_unpadded=moving[:orig_height,:orig_width]
            fixed_unpadded=fixed[:orig_height,:orig_width]
            registered_tissue_unpadded=registered_tissue[:orig_height,:orig_width]
            L2_warp=L2_warp[:orig_height,:orig_width]
            mean_rmse=np.mean(L2_warp)
            intensity_corrected_tissue,intensity_factor=Utils.adjust_intensity(fixed_unpadded,registered_tissue_unpadded)
            before_ncc=Utils.calculate_ncc(fixed_unpadded.ravel(),moving_unpadded.ravel())
            after_ncc=Utils.calculate_ncc(fixed_unpadded.ravel(),registered_tissue_unpadded.ravel())
            writer.writerow([tissue, round_x, before_ncc, after_ncc,intensity_factor,mean_rmse])