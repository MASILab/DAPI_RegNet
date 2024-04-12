from my_utils import Utils
import torch
import os
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm
import matplotlib.pyplot as plt
import glob
import re
import csv
from tifffile import imwrite


original_tissues_path='/nfs2/baos1/rudravg'
round_0_name='GCA112TIA'
os.mkdir(f'/home-local/rudravg/test_DAPI/Registration_QA/{round_0_name}_masked')
os.mkdir(f'/home-local/rudravg/test_DAPI/Registration_QA/{round_0_name}_masked/Registered_Tissues')
os.mkdir(f'/home-local/rudravg/test_DAPI/Registration_QA/{round_0_name}_masked/Intensity_Corrected_Tissues')
os.mkdir(f'/home-local/rudravg/test_DAPI/Registration_QA/{round_0_name}_masked/QA_Images')
save_dir='/home-local/rudravg/test_DAPI/Registration_QA/'+round_0_name+'_masked'
qa_dir='/home-local/rudravg/test_DAPI/Registration_QA/'+round_0_name+'_masked/QA_Images'
model_path='/home-local/rudravg/test_DAPI/epochs/epoch_50/epoch_50.pth'
#Original Tissue files
search_pattern = os.path.join(original_tissues_path, round_0_name + '*')
matching_files = glob.glob(search_pattern)
def extract_round_number(filename):
    match = re.search(r'_ROUND_(\d+)_', filename)
    if match:
        return int(match.group(1))
    else:
        return 0
sorted_files = sorted(matching_files, key=extract_round_number) # List of paths for stored files.
mask_search_pattern = os.path.join(original_tissues_path, 'Retention_Masks', round_0_name + '*')


#Mask files
mask_files = glob.glob(mask_search_pattern)
if mask_files:
    mask_name = mask_files[0]
else:
    mask_name = None

#for i in range(1,len(sorted_files)):
round_0=sorted_files[0]
with open('/home-local/rudravg/test_DAPI/Registration_QA/stuff.csv', mode='a') as file:
    writer=csv.writer(file)
    writer.writerow(['Round','Before NCC','After NCC','Intensity Factor'])
    for i in range(1,len(sorted_files)):
        round_x=sorted_files[i]
        moving,fixed,orig_height,orig_width=Utils.load_images_and_apply_mask(round_0,round_x,mask_name)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model,device=Utils.load_model(model_path)
        registered_tissue,L2_warp=Utils.register_tissues(moving,fixed,model,device)
        moving_unpadded=moving[:orig_height,:orig_width]
        fixed_unpadded=fixed[:orig_height,:orig_width]
        registered_tissue_unpadded=registered_tissue[:orig_height,:orig_width]
        L2_warp=L2_warp[:orig_height,:orig_width]
        before_ncc=Utils.calculate_ncc(fixed_unpadded.ravel(),moving_unpadded.ravel())
        after_ncc=Utils.calculate_ncc(fixed_unpadded.ravel(),registered_tissue_unpadded.ravel())
        print(registered_tissue_unpadded.shape)
        print('Before ncc:',before_ncc)
        print('After ncc:',after_ncc)
        intensity_corrected_image,intensity_factor=Utils.adjust_intensity(fixed_unpadded,registered_tissue_unpadded)
        writer.writerow([round_x[20:], before_ncc[0], after_ncc[0], intensity_factor])
        #Save the registered tissues as tiff files
        imwrite(f'{save_dir}/Registered_Tissues/{round_x[20:]}',registered_tissue_unpadded)
        imwrite(f'{save_dir}/Intensity_Corrected_Tissues/{round_x[20:]}',intensity_corrected_image)

        plt.figure(figsize=(10, 10))
        plt.imshow(L2_warp, cmap='jet',vmin=0, vmax=30)
        plt.colorbar(label='Displacement Magnitude')
        plt.title("L2 norm of composed displacemnt field")
        plt.savefig(f'{qa_dir}/L2_norm_{round_x[20:]}.png')  # Save the L2 norm plot
        
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        # Display the original tissue in the first subplot
        axs[0].imshow(fixed_unpadded, cmap='gray')
        axs[0].set_title('Fixed Tissue')
        axs[0].axis('off')
        # Display the target tissue in the second subplot
        axs[1].imshow(moving_unpadded, cmap='gray')
        axs[1].set_title('Moving Tissue')
        axs[1].axis('off')
        # Display the registered tissue in the third subplot
        axs[2].imshow(registered_tissue_unpadded, cmap='gray')
        axs[2].set_title('Registered Tissue')
        axs[2].axis('off')
        axs[3].imshow(intensity_corrected_image, cmap='gray')
        axs[3].set_title('Intensity Corrected Tissue')
        axs[3].axis('off')

        #Save the plot as a png in QA folder
        plt.savefig(f'{qa_dir}/{round_x[20:]}.png')
