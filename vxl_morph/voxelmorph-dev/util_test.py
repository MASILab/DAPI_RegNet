from my_utils import Utils
import torch
import os
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm
import matplotlib.pyplot as plt

original_tissue='/nfs2/baos1/rudravg/GCA112TIA_DAPI_DAPI_30ms_ROUND_00_initial_reg.tif'
target_tissue='/nfs2/baos1/rudravg/GCA112TIA_DAPI_DAPI_12ms_ROUND_19_initial_reg.tif'
mask='/nfs2/baos1/rudravg/Retention_Masks/GCA112TIA_TISSUE_RETENTION.tif'
model='/home-local/rudravg/test_DAPI/epochs/epoch_50/epoch_50.pth'

fixed,moving=Utils.load_images_and_apply_mask(original_tissue,target_tissue,mask)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model,device=Utils.load_model(model)
registered_tissue=Utils.register_tissues(moving,fixed,model,device)
after_ncc=Utils.calculate_ncc(fixed.ravel(),registered_tissue.ravel())
before_ncc=Utils.calculate_ncc(fixed.ravel(),moving.ravel())
intensity_corrected_image=Utils.adjust_intensity(fixed,registered_tissue)

print('NCC between moving and recon:',before_ncc)
print('NCC between fixed and recon:',after_ncc)
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

# Display the original tissue in the first subplot
axs[0].imshow(fixed, cmap='gray')
axs[0].set_title('Target Tissue')
axs[0].axis('off')

# Display the target tissue in the second subplot
axs[1].imshow(moving, cmap='gray')
axs[1].set_title('Original Tissue')
axs[1].axis('off')

# Display the registered tissue in the third subplot
axs[2].imshow(registered_tissue, cmap='gray')
axs[2].set_title('Registered Tissue')
axs[2].axis('off')

axs[3].imshow(intensity_corrected_image, cmap='gray')
axs[3].set_title('Intensity Corrected Tissue')
axs[3].axis('off')

# Show the plot
plt.show()


#Save the three tissues as png images
plt.imsave('/home-local/rudravg/test_DAPI/combining_patches/target_tissue.png',fixed,cmap='gray')
plt.imsave('/home-local/rudravg/test_DAPI/combining_patches/original_tissue.png',moving,cmap='gray')
plt.imsave('/home-local/rudravg/test_DAPI/combining_patches/registered_tissue.png',registered_tissue,cmap='gray')