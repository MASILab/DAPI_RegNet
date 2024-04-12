from PIL import Image
import numpy as np
import os
import glob
from tqdm import tqdm
from ants_reg import ImageRegistration
import ants

original_dir = '/nfs2/baos1/rudravg/GCA053TIA_TISSUE01*.tif'
output_dir = '/nfs2/baos1/rudravg/'

orig_files = glob.glob(original_dir)
orig_files.sort(key=lambda x: int(x.split('/')[-1].split('_')[-3]))

fixed_file=orig_files[0]

prefix1='registration/registration_'+(fixed_file.split('/')[-1].split('_')[0])+'_'+(fixed_file.split('/')[-1].split('_')[1])+'/'
#os.makedirs(os.path.join(output_dir, prefix1, 'rigid'), exist_ok=True)
#os.makedirs(os.path.join(output_dir, prefix1, 'affine'), exist_ok=True)
print(f'registering {fixed_file}')
os.makedirs(os.path.join(output_dir, prefix1, 'syn'), exist_ok=True)
for file in tqdm(orig_files[1:]):
    moving_file = file
    reg = ImageRegistration(fixed_file, moving_file)
    #rigid=reg.register_rigid()
    #affine=reg.register_affine()
    syn=reg.register_syn_aggro()
    #Save rigid with the file name in the output directory/rigid
    #ants.image_write(rigid, os.path.join(output_dir, prefix1, 'rigid', file.split('/')[-1]))
    #Save affine with the file name in the output directory/affine
    #ants.image_write(affine, os.path.join(output_dir, prefix1, 'affine', file.split('/')[-1]))
    #Save syn with the file name in the output directory/syn
    ants.image_write(syn, os.path.join(output_dir, prefix1, 'syn', file.split('/')[-1]))