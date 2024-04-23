import os
import numpy as np
from PIL import Image
from tqdm import tqdm

text_file="/fs5/p_masi/baos1/rudravg/MXIF/MXIF/Helmsley/MxIF/round_0_dapis.txt"
retention_dir='/nfs2/baos1/rudravg/Retention_Masks'
Image.MAX_IMAGE_PIXELS = None

with open(text_file, 'r') as file:
    lines = file.readlines()


for line in tqdm(lines):
    print(line)
    if 'TISSUE' in line:
        sub_tissue_name=(line.split('/')[4].split('_')[0])+'_'+(line.split('/')[4].split('_')[1])
        sub_name=(line.split('/')[4].split('_')[0])
    else:
        sub_name=(line.split('/')[4].split('_')[0])
        sub_tissue_name=sub_name
    set_name=line.split('/')[1]
    set_dir=f'/fs5/p_masi/baos1/rudravg/MXIF/MXIF/Helmsley/MxIF/{set_name}/{sub_name}/Registered'
    matching_files = [name for name in os.listdir(retention_dir) if sub_tissue_name in name]
    if not matching_files:
        print(f"No mask files for subject {sub_tissue_name}")
        continue
    else:
        mask_file=f'{retention_dir}/{matching_files[0]}'
        mask = Image.open(mask_file)
        orig_size=mask.size
        file_names = [name for name in os.listdir(set_dir) if sub_tissue_name in name]
        first_image = Image.open(os.path.join(set_dir, file_names[0]))
        mask=mask.resize(first_image.size,Image.NEAREST)
        new_size=mask.size
        assert new_size[0] in range(2*orig_size[0]-1, 2*orig_size[0]+2) and new_size[1] in range(2*orig_size[1]-1, 2*orig_size[1]+2), f"Error: check mask for {sub_tissue_name}"
        mask=np.array(mask)
        mask=mask/255.    
        Image.fromarray(mask).save(os.path.join(set_dir,f'{sub_tissue_name}_RetentionMask.tif'))