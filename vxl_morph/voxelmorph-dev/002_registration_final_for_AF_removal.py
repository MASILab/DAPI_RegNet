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
import fnmatch
import matplotlib.pyplot as plt
from my_utils import AllStainUtils
import imageio

mask_dir='/nfs2/baos1/rudravg/Retention_Masks'
tissue_path='/fs5/p_masi/baos1/rudravg/MXIF/MXIF/Helmsley/MxIF/Set01/GCA002ACB/Registered'
tissue_name='GCA002ACB'

if not os.path.exists(os.path.join(tissue_path, 'Deep_Registration')):
    os.mkdir(os.path.join(tissue_path, 'Deep_Registration'))

save_path=os.path.join(tissue_path,'Deep_Registration')

for file in os.listdir(mask_dir):
    if fnmatch.fnmatch(file, '*{}*'.format(tissue_name)):
        mask_path=os.path.join(mask_dir, file)

all_tissues=[]
for file in os.listdir(tissue_path):
    if fnmatch.fnmatch(file, '*.tif'):
        all_tissues.append(os.path.join(tissue_path, file))

def sort_key(file_name):
    match = re.search(r'ROUND_(\d+)', file_name)
    if match:
        return int(match.group(1))
    else:
        return 0

# Sort the file names
sorted_file_names = sorted(all_tissues, key=sort_key)

# Break the sorted file names into sets of 4
rounds = [sorted_file_names[i:i+4] for i in range(0, len(sorted_file_names), 4)]

# Store each set in a different array and ensure 'DAPI' is the first element
for i, round in enumerate(rounds):
    round.sort(key=lambda x: 'DAPI' not in x)
    globals()[f'Round_{i+1}'] = round

DAPI_round_0=rounds[0][0]

model_path='/home-local/rudravg/test_DAPI/epochs/epoch_50/epoch_50.pth'
model,device=Utils.load_model(model_path)
device='cuda:0'
model.to(device)
model.eval()

with open(os.path.join(save_path,'output.txt'), 'a') as f:
        round0, roundx, stain1_recon, stain2_recon, stain3_recon, mask = AllStainUtils.load_all_stains_and_dapi(DAPI_round_0, DAPI_round_0, rounds[0][1], rounds[0][2], rounds[0][3], mask_path)
        f.write(f'DAPI_roundx: {DAPI_round_0}\n')
        f.write(f'stain1: {rounds[0][1]}\n')
        f.write(f'stain2: {rounds[0][2]}\n')
        f.write(f'stain3: {rounds[0][3]}\n')
        f.write(f'mask_path: {mask_path}\n')
        f.write('\n')  
        Image.fromarray(roundx).save(f"{save_path}/{os.path.basename(DAPI_round_0)}")
        Image.fromarray(stain1_recon).save(f"{save_path}/{os.path.basename(rounds[0][1])}")
        Image.fromarray(stain2_recon).save(f"{save_path}/{os.path.basename(rounds[0][2])}")
        Image.fromarray(stain3_recon).save(f"{save_path}/{os.path.basename(rounds[0][3])}")


with open(os.path.join(save_path,'output.txt'), 'a') as f:
    for round in tqdm(rounds[1:]):
        DAPI_roundx = round[0]
        stain1 = round[1]
        stain2 = round[2]
        stain3 = round[3]
        round0, roundx, stain1_recon, stain2_recon, stain3_recon, mask = AllStainUtils.load_all_stains_and_dapi(DAPI_round_0, DAPI_roundx, stain1, stain2, stain3, mask_path)
        (recon_tissue, recon_stain1, recon_stain2, recon_stain3) = AllStainUtils.do_registration(
            fixed=round0, 
            moving_img=roundx, 
            stain1=stain1_recon,
            stain2=stain2_recon, 
            stain3=stain3_recon, 
            model=model,
            device=device
        )
        f.write(f'DAPI_roundx: {DAPI_roundx}\n')
        f.write(f'stain1: {stain1}\n')
        f.write(f'stain2: {stain2}\n')
        f.write(f'stain3: {stain3}\n')
        f.write(f'mask_path: {mask_path}\n')
        f.write('\n')  
        Image.fromarray(recon_tissue).save(f"{save_path}/{os.path.basename(DAPI_roundx)}")
        Image.fromarray(recon_stain1).save(f"{save_path}/{os.path.basename(stain1)}")
        Image.fromarray(recon_stain2).save(f"{save_path}/{os.path.basename(stain2)}")
        Image.fromarray(recon_stain3).save(f"{save_path}/{os.path.basename(stain3)}")

        
