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
from skimage.util import view_as_windows
from math import ceil

Image.MAX_IMAGE_PIXELS = None


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
    def load_images_and_no_mask(moving, fixed, mask):
        moving=np.array(Image.open(moving))
        fixed=np.array(Image.open(fixed))
        mask=np.array(Image.open(mask))
        mask = (mask > 0).astype(int)
        mask = mask/255.
        moving_tissue_masked = moving /255.
        fixed_tissue_masked = fixed /255.
        #moving_tissue_masked = moving * mask
        #fixed_tissue_masked = fixed * mask
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
    
class AllStainUtils:
    @staticmethod
    def load_all_stains_and_dapi(dapi_r0,dapi_rx,stain1,stain2,stain3,mask):
        """
        This function does 3 things. Load the images, downscale them to be the same size as the mask and apply the mask.
        """
        # Load the images
        dapi_r0 = Image.open(dapi_r0)
        dapi_rx = Image.open(dapi_rx)
        stain1 = Image.open(stain1)
        stain2 = Image.open(stain2)
        stain3 = Image.open(stain3)
        mask = Image.open(mask)

        # Resize the images to be the same size as the mask
        dapi_r0 = dapi_r0.resize(mask.size)
        dapi_rx = dapi_rx.resize(mask.size)
        stain1 = stain1.resize(mask.size)
        stain2 = stain2.resize(mask.size)
        stain3 = stain3.resize(mask.size)

        # Convert images to numpy arrays
        dapi_r0 = np.array(dapi_r0)
        dapi_rx = np.array(dapi_rx)
        stain1 = np.array(stain1)
        stain2 = np.array(stain2)
        stain3 = np.array(stain3)
        mask = np.array(mask)
        print(np.unique(mask))

        # Apply the mask
        mask = (mask > 0).astype(int)
        print(np.unique(mask))
        dapi_r0 = (dapi_r0 * mask)/255.
        dapi_rx = (dapi_rx * mask)/255.
        stain1 = (stain1 * mask)/255.
        stain2 = (stain2 * mask)/255.
        stain3 = (stain3 * mask)/255.

        return dapi_r0, dapi_rx, stain1, stain2, stain3, mask

    @staticmethod
    def do_registration(moving_img, fixed,stain1,stain2,stain3, model, device):
        """
        This function does the registration of the moving image to the fixed image using the model provided. It also
        applies the same registratin to the stains provided.
        """
        block_size = (512, 512)
        num_blocks_x = moving_img.shape[0] // block_size[0]
        num_blocks_y = moving_img.shape[1] // block_size[1]
        moving= moving_img[:num_blocks_x * block_size[0], :num_blocks_y * block_size[1]]
        fixed= fixed[:num_blocks_x * block_size[0], :num_blocks_y * block_size[1]]
        stain1= stain1[:num_blocks_x * block_size[0], :num_blocks_y * block_size[1]]
        stain2= stain2[:num_blocks_x * block_size[0], :num_blocks_y * block_size[1]]
        stain3= stain3[:num_blocks_x * block_size[0], :num_blocks_y * block_size[1]]
        moving_tissue_blocks = view_as_blocks(moving, block_shape=block_size)
        fixed_tissue_blocks = view_as_blocks(fixed, block_shape=block_size)
        stain1_blocks= view_as_blocks(stain1, block_shape=block_size)
        stain2_blocks= view_as_blocks(stain2, block_shape=block_size)
        stain3_blocks= view_as_blocks(stain3, block_shape=block_size)
        pred_blocks_tissue = []
        pred_blocks_stain1=[]
        pred_blocks_stain2=[]
        pred_blocks_stain3=[]

        for i in range(moving_tissue_blocks.shape[0]):
            row_blocks_tissues = []
            row_blocks_stain1 = []
            row_blocks_stain2 = []
            row_blocks_stain3 = []

            for j in range(moving_tissue_blocks.shape[1]):
                moving_block = moving_tissue_blocks[i, j]
                fixed_block = fixed_tissue_blocks[i, j]
                stain1_block = stain1_blocks[i,j]
                stain2_block = stain2_blocks[i,j]
                stain3_block = stain3_blocks[i,j]

                moving_block = moving_block[np.newaxis, ..., np.newaxis]
                fixed_block = fixed_block[np.newaxis, ..., np.newaxis]
                stain1_block = stain1_block[np.newaxis, ..., np.newaxis]
                stain2_block = stain2_block[np.newaxis, ..., np.newaxis]
                stain3_block = stain3_block[np.newaxis, ..., np.newaxis]

                moving_block = torch.from_numpy(moving_block).to(device).float().permute(0,3,1,2)
                fixed_block = torch.from_numpy(fixed_block).to(device).float().permute(0,3,1,2)
                stain1_block = torch.from_numpy(stain1_block).to(device).float().permute(0,3,1,2)
                stain2_block = torch.from_numpy(stain2_block).to(device).float().permute(0,3,1,2)
                stain3_block = torch.from_numpy(stain3_block).to(device).float().permute(0,3,1,2)

                fwd_pred,fwd_pred_field = model(moving_block,fixed_block ,registration=True)
                stain1_pred=model.transformer(stain1_block,fwd_pred_field)
                stain2_pred=model.transformer(stain2_block,fwd_pred_field)
                stain3_pred=model.transformer(stain3_block,fwd_pred_field)

                row_blocks_tissues.append(fwd_pred.detach().cpu().numpy())
                row_blocks_stain1.append(stain1_pred.detach().cpu().numpy())
                row_blocks_stain2.append(stain2_pred.detach().cpu().numpy())
                row_blocks_stain3.append(stain3_pred.detach().cpu().numpy())

            pred_blocks_tissue.append(row_blocks_tissues)
            pred_blocks_stain1.append(row_blocks_stain1)
            pred_blocks_stain2.append(row_blocks_stain2)
            pred_blocks_stain3.append(row_blocks_stain3)

        reconstructed_tissue = np.block(pred_blocks_tissue)
        reconstructed_stain1 = np.block(pred_blocks_stain1)
        reconstructed_stain2 = np.block(pred_blocks_stain2)
        reconstructed_stain3 = np.block(pred_blocks_stain3)

        reconstructed_tissue = reconstructed_tissue.squeeze().squeeze()
        reconstructed_stain1 = reconstructed_stain1.squeeze().squeeze()
        reconstructed_stain2 = reconstructed_stain2.squeeze().squeeze()
        reconstructed_stain3 = reconstructed_stain3.squeeze().squeeze()

        return reconstructed_tissue, reconstructed_stain1, reconstructed_stain2, reconstructed_stain3
    
    @staticmethod
    def load_tissues_for_overlap(moving,fixed,mask):
        moving=np.array(Image.open(moving))
        fixed=np.array(Image.open(fixed))
        mask=np.array(Image.open(mask))
        moving=(moving*mask)/255.
        fixed=(fixed*mask)/255.
        # Calculate the required padding size
        patch_size = 1024
        overlap = 200

        n_patches_height = ceil((moving.shape[0] - overlap) / (patch_size - overlap))
        n_patches_width = ceil((moving.shape[1] - overlap) / (patch_size - overlap))

        # Calculate the required padding size
        pad_height = n_patches_height * (patch_size - overlap) + overlap - moving.shape[0]
        pad_width = n_patches_width * (patch_size - overlap) + overlap - moving.shape[1]

        # Pad the images
        moving = np.pad(moving, ((0, pad_height), (0, pad_width)), mode='constant')
        fixed = np.pad(fixed, ((0, pad_height), (0, pad_width)), mode='constant')

        return moving, fixed


    
    @staticmethod
    def register_tissues_with_overlap(moving, fixed, model, device):
        block_size = (1024, 1024)
        overlap = 200
        stride = (block_size[0] - overlap, block_size[1] - overlap)
        height, width = moving.shape

        # Accumulator arrays for averaging overlapping areas
        full_tissue = np.zeros_like(moving, dtype=np.float32)
        full_field = np.zeros_like(moving, dtype=np.float32)
        count_map = np.zeros_like(moving, dtype=np.float32)

        num_blocks_x = (width - overlap) // stride[1] + 1
        num_blocks_y = (height - overlap) // stride[0] + 1

        for i in range(num_blocks_y):
            for j in range(num_blocks_x):
                y_start = i * stride[0]
                x_start = j * stride[1]
                y_end = min(y_start + block_size[0], height)
                x_end = min(x_start + block_size[1], width)

                moving_block = moving[y_start:y_end, x_start:x_end]
                fixed_block = fixed[y_start:y_end, x_start:x_end]

                moving_block = moving_block[np.newaxis, ..., np.newaxis]
                fixed_block = fixed_block[np.newaxis, ..., np.newaxis]
                if moving_block.shape!=(1, 1024, 1024, 1):
                    continue
                moving_block = torch.from_numpy(moving_block).to(device).float().permute(0, 3, 1, 2)
                fixed_block = torch.from_numpy(fixed_block).to(device).float().permute(0, 3, 1, 2)

                fwd_pred = model(moving_block, fixed_block, registration=True)


                # Update full image and field accumulators
                full_tissue[y_start:y_end, x_start:x_end] += fwd_pred[0].detach().cpu().numpy().squeeze()
                count_map[y_start:y_end, x_start:x_end] += 1

        # Averaging the accumulated values
        full_tissue /= count_map

        return full_tissue






    

    
