import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import torch
from my_utils import Utils
from skimage.util import view_as_blocks
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model='/home-local/rudravg/test_DAPI/epochs/epoch_35/epoch_35.pth'
best_model=vxm.networks.VxmDense.load(model, device)
best_model.to(device)
best_model.eval()

def registerTissues(original_tissue,target_tissue,model,device):
    block_size = (512, 512)
    num_blocks_x = original_tissue.shape[0] // block_size[0]
    num_blocks_y = original_tissue.shape[1] // block_size[1]
    original_tissue_cropped = original_tissue[:num_blocks_x * block_size[0], :num_blocks_y * block_size[1]]
    target_tissue_cropped = target_tissue[:num_blocks_x * block_size[0], :num_blocks_y * block_size[1]]
    original_tissue_blocks = view_as_blocks(original_tissue_cropped, block_shape=block_size)
    target_tissue_blocks = view_as_blocks(target_tissue_cropped, block_shape=block_size)
    pred_blocks = []

    for i in range (original_tissue_blocks.shape[0]):
        row_blocks = []
        for j in range (original_tissue_blocks.shape[1]):
            original_tissue_block = original_tissue_blocks[i, j]
            target_tissue_block = target_tissue_blocks[i, j]
            original_tissue_block = original_tissue_block[np.newaxis, ..., np.newaxis]
            target_tissue_block = target_tissue_block[np.newaxis, ..., np.newaxis]
            original_tissue_block=torch.from_numpy(original_tissue_block).to(device).float().permute(0,3,1,2)
            target_tissue_block=torch.from_numpy(target_tissue_block).to(device).float().permute(0,3,1,2)
            pred = model(original_tissue_block, target_tissue_block)
            row_blocks.append(pred[0].detach().cpu().numpy())
        pred_blocks.append(row_blocks)
    reconstructed_tissue = np.block(pred_blocks)
    reconstructed_tissue = reconstructed_tissue.squeeze().squeeze()
    reconstructed_tissue_padded = np.zeros(original_tissue.shape)
    reconstructed_tissue_padded[:reconstructed_tissue.shape[0], :reconstructed_tissue.shape[1]] = reconstructed_tissue
    return reconstructed_tissue_padded

def calculate_ncc(array1, array2):
    array1 = (array1 - np.mean(array1)) / (np.std(array1) * len(array1))
    array2 = (array2 - np.mean(array2)) / (np.std(array2))
    ncc = np.correlate(array1, array2)
    return ncc

def load_and_mask(original_t,target_t,mask):
    original_tissue=np.array(Image.open(original_t))
    target_tissue=np.array(Image.open(target_t))
    mask=np.array(Image.open(mask))
    mask = (mask > 0).astype(int)
    original_tissue=original_tissue/255.
    target_tissue=target_tissue/255.
    mask=mask/255.
    original_tissue=mask*original_tissue
    target_tissue=mask*target_tissue
    return original_tissue,target_tissue

original_tissue='/nfs2/baos1/rudravg/GCA112TIA_DAPI_DAPI_30ms_ROUND_00_initial_reg.tif'
target_tissue='/nfs2/baos1/rudravg/GCA112TIA_DAPI_DAPI_12ms_ROUND_19_initial_reg.tif'
mask='/nfs2/baos1/rudravg/Retention_Masks/GCA112TIA_TISSUE_RETENTION.tif'

#original_tissue=np.array(Image.open(original_tissue))
#target_tissue=np.array(Image.open(target_tissue))
#mask=np.array(Image.open(mask))
#mask = (mask > 0).astype(int)

#original_tissue=original_tissue/255.
#target_tissue=target_tissue/255.
#mask=mask/255.
#original_tissue=mask*original_tissue
#target_tissue=mask*target_tissue

original_tissue,target_tissue=load_and_mask(original_tissue,target_tissue,mask)
reconstructed_tissue=registerTissues(original_tissue,target_tissue,best_model,device)
orig_recon=calculate_ncc(original_tissue.ravel(),reconstructed_tissue.ravel())
target_recon=calculate_ncc(target_tissue.ravel(),reconstructed_tissue.ravel())
#Save the original, reconstructed and target tissues as tiff files
Image.fromarray((original_tissue*255).astype(np.uint8)).save('/home-local/rudravg/test_DAPI/Registration_QA/GCA112TIA_masked/Apr14_tissue_reg_check/original_tissue.tif')
Image.fromarray((reconstructed_tissue*255).astype(np.uint8)).save('/home-local/rudravg/test_DAPI/Registration_QA/GCA112TIA_masked/Apr14_tissue_reg_check/reconstructed_tissue.tif')
Image.fromarray((target_tissue*255).astype(np.uint8)).save('/home-local/rudravg/test_DAPI/Registration_QA/GCA112TIA_masked/Apr14_tissue_reg_check/target_tissue.tif')

print('NCC between original and reconstructed tissue:',orig_recon)
print('NCC between target and reconstructed tissue:',target_recon)

