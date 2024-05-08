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

class Utils_v2:
    @staticmethod
    def load_tissues_for_overlap(tissues,mask):
        """"
        Args:
        tissues: tissue path
        mask: Path to the mask image
        Returns: Padded tissues, height and width of the mask
        """
        mask=np.array(Image.open(mask))
        height, width = mask.shape
        patch_size = 1024
        overlap = 200
        n_patches_height = ceil((mask.shape[0] - overlap) / (patch_size - overlap))
        n_patches_width = ceil((mask.shape[1] - overlap) / (patch_size - overlap))
        pad_height = n_patches_height * (patch_size - overlap) + overlap - mask.shape[0]
        pad_width = n_patches_width * (patch_size - overlap) + overlap - mask.shape[1]
        tissue=np.array(Image.open(tissues))
        tissue=(tissue*mask)/255.
        tissue = np.pad(tissue, ((0, pad_height), (0, pad_width)), mode='constant')

        return tissue,height,width
    
    @staticmethod
    def register_tissues_with_overlap(moving, fixed, model, device,stain1=None):
        """
        Register two images with overlap
        Args:
        moving: moving image
        fixed: fixed image
        model: Model
        device: Device
        stain: Stain image
        Returns: Registered tissue and stain image
        """
        block_size = (1024, 1024)
        overlap = 200
        stride = (block_size[0] - overlap, block_size[1] - overlap)
        height, width = moving.shape

        # Accumulator arrays for averaging overlapping areas
        full_tissue = np.zeros_like(moving, dtype=np.float32)
        stain1 = np.zeros_like(moving, dtype=np.float32)
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
                stain_block = stain1[y_start:y_end, x_start:x_end]

                moving_block = moving_block[np.newaxis, ..., np.newaxis]
                fixed_block = fixed_block[np.newaxis, ..., np.newaxis]
                stain_block = stain_block[np.newaxis, ..., np.newaxis]
                if moving_block.shape!=(1, 1024, 1024, 1):
                    continue
                moving_block = torch.from_numpy(moving_block).to(device).float().permute(0, 3, 1, 2)
                fixed_block = torch.from_numpy(fixed_block).to(device).float().permute(0, 3, 1, 2)

                fwd_pred,fwd_pred_field = model(moving_block, fixed_block, registration=True)
                stain1_block = model.transformer(stain_block, fwd_pred_field)

                # Update full image and field accumulators
                full_tissue[y_start:y_end, x_start:x_end] += fwd_pred.detach().cpu().numpy().squeeze()
                stain1[y_start:y_end, x_start:x_end] += stain1_block.detach().cpu().numpy().squeeze()
                count_map[y_start:y_end, x_start:x_end] += 1

        # Averaging the accumulated values
        full_tissue /= count_map
        stain1 /= count_map

        return full_tissue,stain1
    
    @staticmethod
    def register_multiple_tissues_with_overlap(moving, fixed, model, device,stain1=None,stain2=None,stain3=None,stain4=None,stain5=None,stain6=None):
        """
        Register two images and corresponding stains with overlap
        Args:
        moving: moving image
        fixed: fixed image
        stain1: First stain image
        stain2: Second stain image
        stain3: Third stain image
        model: Model
        device: Device
        stain: Stain image
        Returns: Registered tissue and stain image
        """
        block_size = (1024, 1024)
        overlap = 200
        stride = (block_size[0] - overlap, block_size[1] - overlap)
        height, width = moving.shape

        # Accumulator arrays for averaging overlapping areas
        full_tissue = np.zeros_like(moving, dtype=np.float32)
        stain1_accumulator = np.zeros_like(moving, dtype=np.float32) if stain1 is not None else None
        stain2_accumulator = np.zeros_like(moving, dtype=np.float32) if stain2 is not None else None
        stain3_accumulator = np.zeros_like(moving, dtype=np.float32) if stain3 is not None else None
        stain4_accumulator = np.zeros_like(moving, dtype=np.float32) if stain4 is not None else None
        stain5_accumulator = np.zeros_like(moving, dtype=np.float32) if stain5 is not None else None
        stain6_accumulator = np.zeros_like(moving, dtype=np.float32) if stain6 is not None else None
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
                stain1_block = stain1[y_start:y_end, x_start:x_end] if stain1 is not None else None
                stain2_block = stain2[y_start:y_end, x_start:x_end] if stain2 is not None else None
                stain3_block = stain3[y_start:y_end, x_start:x_end] if stain3 is not None else None
                stain4_block = stain4[y_start:y_end, x_start:x_end] if stain4 is not None else None
                stain5_block = stain5[y_start:y_end, x_start:x_end] if stain5 is not None else None
                stain6_block = stain6[y_start:y_end, x_start:x_end] if stain6 is not None else None

                moving_block = moving_block[np.newaxis, ..., np.newaxis]
                fixed_block = fixed_block[np.newaxis, ..., np.newaxis]
                if stain1_block is not None:
                    stain1_block = stain1_block[np.newaxis, ..., np.newaxis]
                if stain2_block is not None:
                    stain2_block = stain2_block[np.newaxis, ..., np.newaxis]
                if stain3_block is not None:
                    stain3_block = stain3_block[np.newaxis, ..., np.newaxis]
                if stain4_block is not None:
                    stain4_block = stain4_block[np.newaxis, ..., np.newaxis]
                if stain5_block is not None:
                    stain5_block = stain5_block[np.newaxis, ..., np.newaxis]
                if stain6_block is not None:
                    stain6_block = stain6_block[np.newaxis, ..., np.newaxis]
                if moving_block.shape!=(1, 1024, 1024, 1):
                    continue
                moving_block = torch.from_numpy(moving_block).to(device).float().permute(0, 3, 1, 2)
                fixed_block = torch.from_numpy(fixed_block).to(device).float().permute(0, 3, 1, 2)

                fwd_pred,fwd_pred_field = model(moving_block, fixed_block, registration=True)
                if stain1_block is not None:
                    stain1_block = torch.from_numpy(stain1_block).to(device).float().permute(0, 3, 1, 2)
                    stain1_block = model.transformer(stain1_block, fwd_pred_field)
                    stain1_accumulator[y_start:y_end, x_start:x_end] += stain1_block.detach().cpu().numpy().squeeze()
                if stain2_block is not None:
                    stain2_block = torch.from_numpy(stain2_block).to(device).float().permute(0, 3, 1, 2)
                    stain2_block = model.transformer(stain2_block, fwd_pred_field)
                    stain2_accumulator[y_start:y_end, x_start:x_end] += stain2_block.detach().cpu().numpy().squeeze()
                if stain3_block is not None:
                    stain3_block = torch.from_numpy(stain3_block).to(device).float().permute(0, 3, 1, 2)
                    stain3_block = model.transformer(stain3_block, fwd_pred_field)
                    stain3_accumulator[y_start:y_end, x_start:x_end] += stain3_block.detach().cpu().numpy().squeeze()
                if stain4_block is not None:
                    stain4_block = torch.from_numpy(stain4_block).to(device).float().permute(0, 3, 1, 2)
                    stain4_block = model.transformer(stain4_block, fwd_pred_field)
                    stain4_accumulator[y_start:y_end, x_start:x_end] += stain4_block.detach().cpu().numpy().squeeze()
                if stain5_block is not None:
                    stain5_block = torch.from_numpy(stain5_block).to(device).float().permute(0, 3, 1, 2)
                    stain5_block = model.transformer(stain5_block, fwd_pred_field)
                    stain5_accumulator[y_start:y_end, x_start:x_end] += stain5_block.detach().cpu().numpy().squeeze()
                if stain6_block is not None:
                    stain6_block = torch.from_numpy(stain6_block).to(device).float().permute(0, 3, 1, 2)
                    stain6_block = model.transformer(stain6_block, fwd_pred_field)
                    stain6_accumulator[y_start:y_end, x_start:x_end] += stain6_block.detach().cpu().numpy().squeeze()
                # Update full image and field accumulators
                full_tissue[y_start:y_end, x_start:x_end] += fwd_pred.detach().cpu().numpy().squeeze()
                count_map[y_start:y_end, x_start:x_end] += 1

        # Averaging the accumulated values
        full_tissue /= count_map
        if stain6 is not None:
            stain1_accumulator /= count_map
            stain2_accumulator /= count_map
            stain3_accumulator /= count_map
            stain4_accumulator /= count_map
            stain5_accumulator /= count_map
            stain6_accumulator /= count_map
            return full_tissue,stain1_accumulator,stain2_accumulator,stain3_accumulator,stain4_accumulator,stain5_accumulator,stain6_accumulator
        elif stain5 is not None:
            stain1_accumulator /= count_map
            stain2_accumulator /= count_map
            stain3_accumulator /= count_map
            stain4_accumulator /= count_map
            stain5_accumulator /= count_map
            return full_tissue,stain1_accumulator,stain2_accumulator,stain3_accumulator,stain4_accumulator,stain5_accumulator
        elif stain4 is not None:
            stain1_accumulator /= count_map
            stain2_accumulator /= count_map
            stain3_accumulator /= count_map
            stain4_accumulator /= count_map
            return full_tissue,stain1_accumulator,stain2_accumulator,stain3_accumulator,stain4_accumulator
        elif stain3 is not None:
            stain1_accumulator /= count_map
            stain2_accumulator /= count_map
            stain3_accumulator /= count_map
            return full_tissue,stain1_accumulator,stain2_accumulator,stain3_accumulator
        elif stain2 is not None:
            stain1_accumulator /= count_map
            stain2_accumulator /= count_map
            return full_tissue,stain1_accumulator,stain2_accumulator
        elif stain1 is not None:
            stain1_accumulator /= count_map
            return full_tissue,stain1_accumulator
        else:
            return full_tissue
    
    @staticmethod
    def generate_average_DAPI(dir_path):
        pass

    @staticmethod
    def load_model(model_path):
        """"
        Args:
        model_path: Path to the model
        Returns: Model and device
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = vxm.networks.VxmDense.load(model_path, device)
        model.to(device)
        model.eval()
        return model,device
    
    @staticmethod
    def calculate_ncc(array1, array2):
        """
        Calculate normalized cross correlation
        Args:
        array1: 1D array of your image1. Use np.rave() to convert 2D image to 1D
        array2: 1D array of your image2.
        Returns: Normalized cross correlation
        """
        array1 = (array1 - np.mean(array1)) / (np.std(array1) * len(array1))
        array2 = (array2 - np.mean(array2)) / (np.std(array2))
        ncc = np.correlate(array1, array2)
        return ncc
