import numpy as np
import matplotlib.pyplot as plt
from skimage.util import view_as_blocks
import torch
import matplotlib.pyplot as plt
import os
from PIL import Image
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm
from scipy.ndimage import map_coordinates
from skimage import exposure


class Utils:
    @staticmethod
    def adjust_intensity(original_image, target_image):
        original_image = original_image.astype(float)
        target_image = target_image.astype(float)
        ratio = np.where(target_image != 0, original_image / target_image, 0)
        counts, bins = np.histogram(ratio[ratio != 0].ravel(), bins=255)
        max_count_index = np.argmax(counts)
        start = bins[max_count_index]
        end = bins[max_count_index + 1]
        ratios_in_bin = ratio[(ratio > start) & (ratio < end)]
        factor = ratios_in_bin.mean()
        adjusted_target_image = target_image * factor
        adjusted_target_image_new = np.minimum(adjusted_target_image, original_image.max())
        return adjusted_target_image_new,factor
    @staticmethod
    def adjust_intensity_histogram(original_image, target_image):
        # Convert the images to float
        original_image = original_image.astype(float)
        target_image = target_image.astype(float)

        # Perform histogram matching
        matched_target_image = exposure.match_histograms(target_image, original_image)

        # Calculate the factor
        factor = np.mean(original_image) / np.mean(matched_target_image) if np.mean(matched_target_image) != 0 else 0

        return matched_target_image, factor
    
    @staticmethod
    def register_tissues(moving, fixed, model, device):
        block_size = (512, 512)
#        num_blocks_x = moving.shape[0] // block_size[0]
#        num_blocks_y = moving.shape[1] // block_size[1]
#        original_tissue_cropped = moving[:num_blocks_x * block_size[0], :num_blocks_y * block_size[1]]
#        fixed_cropped = fixed[:num_blocks_x * block_size[0], :num_blocks_y * block_size[1]]
        moving_tissue_blocks = view_as_blocks(moving, block_shape=block_size)
        fixed_tissue_blocks = view_as_blocks(fixed, block_shape=block_size)
        pred_blocks_tissue = []
        pred_blocks_field=[]

        for i in range(moving_tissue_blocks.shape[0]):
            row_blocks_tissues = []
            row_blocks_field = []
            for j in range(moving_tissue_blocks.shape[1]):
                moving_block = moving_tissue_blocks[i, j]
                fixed_block = fixed_tissue_blocks[i, j]
                moving_block = moving_block[np.newaxis, ..., np.newaxis]
                fixed_block = fixed_block[np.newaxis, ..., np.newaxis]
                moving_block = torch.from_numpy(moving_block).to(device).float().permute(0,3,1,2)
                fixed_block = torch.from_numpy(fixed_block).to(device).float().permute(0,3,1,2)
                fwd_pred = model(moving_block,fixed_block ,registration=True)
                inv_pred = model(fixed_block,moving_block, registration=True)
                composite_field = Utils.combine_displacement_fields(fwd_pred[1].detach().cpu().numpy().squeeze(), inv_pred[1].detach().cpu().numpy().squeeze())
                L2_norm_combined = np.sqrt(composite_field[0]**2 + composite_field[1]**2)
                row_blocks_tissues.append(fwd_pred[0].detach().cpu().numpy())
                row_blocks_field.append(L2_norm_combined)
            pred_blocks_tissue.append(row_blocks_tissues)
            pred_blocks_field.append(row_blocks_field)

        reconstructed_tissue = np.block(pred_blocks_tissue)
        composed_warp = np.block(pred_blocks_field)
        reconstructed_tissue = reconstructed_tissue.squeeze().squeeze()
        return reconstructed_tissue,composed_warp
    
    @staticmethod
    def load_images_and_apply_mask(moving, fixed, mask):
        moving=np.array(Image.open(moving))
        fixed=np.array(Image.open(fixed))
        mask=np.array(Image.open(mask))
        mask = (mask > 0).astype(int)
        mask = mask/255.
        #moving_tissue_masked = moving /255.
        #fixed_tissue_masked = fixed /255.
        moving_tissue_masked = moving * mask
        fixed_tissue_masked = fixed * mask
        original_tissue_padding = ((0, 512 - moving.shape[0] % 512), (0, 512 - moving.shape[1] % 512))
        original_height=moving.shape[0]
        original_width=moving.shape[1]
        fixed_tissue = np.pad(fixed_tissue_masked, original_tissue_padding, mode='constant')
        moving_tissue = np.pad(moving_tissue_masked, original_tissue_padding, mode='constant')
        return moving_tissue, fixed_tissue,original_height,original_width
    
    @staticmethod
    def load_model(model_path):
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        model = vxm.networks.VxmDense.load(model_path, device)
        model.to(device)
        model.eval()
        return model,device
    
    @staticmethod
    def calculate_ncc(array1, array2):
        array1 = (array1 - np.mean(array1)) / (np.std(array1) * len(array1))
        array2 = (array2 - np.mean(array2)) / (np.std(array2))
        ncc = np.correlate(array1, array2)
        return ncc
    
    @staticmethod
    def combine_displacement_fields(D1, D2):
        assert D1.shape == D2.shape, "Displacement fields must have the same shape"
        
        D_combined = np.zeros_like(D1)
        
        _, height, width = D1.shape
        
        X, Y = np.meshgrid(np.arange(width), np.arange(height))
        
        X_displaced = X + D1[0]
        Y_displaced = Y + D1[1]
        
        X_displaced_clipped = np.clip(X_displaced, 0, width - 1)
        Y_displaced_clipped = np.clip(Y_displaced, 0, height - 1)
        
        D2_x_interpolated = map_coordinates(D2[0], [Y_displaced_clipped, X_displaced_clipped], order=1)
        D2_y_interpolated = map_coordinates(D2[1], [Y_displaced_clipped, X_displaced_clipped], order=1)
        
        D_combined[0] = D1[0] + D2_x_interpolated
        D_combined[1] = D1[1] + D2_y_interpolated
        
        return D_combined
    

    
