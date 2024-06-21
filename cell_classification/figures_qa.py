import pandas as pd
from PIL import Image
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import cv2
from skimage import segmentation


Image.MAX_IMAGE_PIXELS = None

unregistered_csv='GCA020TIB_unregistered_classified.csv'
registered_csv='GCA020TIB_registered_classified.csv'

unregistered_df=pd.read_csv(unregistered_csv)
registered_df=pd.read_csv(registered_csv)

class_labels = [col for col in registered_df.columns if col.startswith("final_")]

analysis_df = pd.DataFrame({
    'Instance': unregistered_df['Instances'],  # Assuming '' is the coInstanceslumn name in your CSV
    'Gain': False,
    'Loss': False,
    'Change': False,
    'Before Classification': '',
    'After Classification': ''
})

# Iterate through each row to determine gain, loss, and change
for index, (reg_row, unreg_row) in enumerate(zip(registered_df.iterrows(), unregistered_df.iterrows())):
    reg_classes = [label for label in class_labels if reg_row[1][label]]
    unreg_classes = [label for label in class_labels if unreg_row[1][label]]

    # Determine before and after classification
    before_classification = unreg_classes[0] if unreg_classes else 'Undefined'
    after_classification = reg_classes[0] if reg_classes else 'Undefined'
    
    analysis_df.at[index, 'Before Classification'] = before_classification
    analysis_df.at[index, 'After Classification'] = after_classification

    # Determine gain, loss, and change
    if unreg_row[1]['Undefined'] and reg_classes:
        analysis_df.at[index, 'Gain'] = True
    elif reg_row[1]['Undefined'] and unreg_classes:
        analysis_df.at[index, 'Loss'] = True
    elif unreg_classes and reg_classes and before_classification != after_classification:
        analysis_df.at[index, 'Change'] = True

indexes_gained=analysis_df[(analysis_df['Before Classification']=='Undefined') & (analysis_df['After Classification']=='final_Progenitor')].index

#Losses for Enteroendocrine
indexes_lost=analysis_df[(analysis_df['Before Classification']=='final_Progenitor') & (analysis_df['After Classification']=='Undefined')].index

#Changes for Enteroendocrine
indexes_changed=analysis_df[(analysis_df['Before Classification']=='final_Progenitor') & (analysis_df['After Classification']!='final_Progenitor') & (analysis_df['After Classification']!='Undefined')].index

tissue_name='GCA033TIB_TISSUE02'
#GCA020TIB_TISSUE03 for Fibroblast and Enterocytes
mask_path=f'/fs5/p_masi/rudravg/MxIF_Vxm_Registered_V2/{tissue_name}/mask.tif'
mask=np.array(Image.open(mask_path))


marker_list = ['CD11B','CD20','CD3D','CD45','CD4','CD68','CD8','CGA','LYSOZYME','NAKATPASE','PANCK','SMA','SOX9','VIMENTIN','OLFM4']

registered_marker_path=f'/fs5/p_masi/rudravg/MxIF_Vxm_Registered_V2/{tissue_name}/AF_Removed/'
unregistered_marker_path=f'/fs5/p_masi/rudravg/MxIF_Vxm_Registered_V2/{tissue_name}/Unregistered/AF_Removed/'

#Find all file names in registered marker_path which have the marker name from the marker list, compare it after capitalizing the file paths
registered_marker_files = [file for file in os.listdir(registered_marker_path) 
                        if any(marker.upper() in file.upper() for marker in marker_list) 
                        and "BAD" not in file.upper()]
#Sort it based on the marker list
registered_marker_files = sorted(registered_marker_files, key=lambda x: [marker.upper() in x.upper() for marker in marker_list], reverse=True)

CD11B_registered=os.path.join(registered_marker_path,registered_marker_files[0])
CD11B_registered=np.uint8(np.array(Image.open(CD11B_registered)))

CD11B_unregistered=os.path.join(unregistered_marker_path,registered_marker_files[0])
CD11B_unregistered=np.uint8(np.array(Image.open(CD11B_unregistered)))

CD20_registered=os.path.join(registered_marker_path,registered_marker_files[1]) 
CD20_registered=np.uint8(np.array(Image.open(CD20_registered)))

CD20_unregistered=os.path.join(unregistered_marker_path,registered_marker_files[1])
CD20_unregistered=np.uint8(np.array(Image.open(CD20_unregistered)))

CD3D_registered=os.path.join(registered_marker_path,registered_marker_files[2])
CD3D_registered=np.uint8(np.array(Image.open(CD3D_registered)))

CD3D_unregistered=os.path.join(unregistered_marker_path,registered_marker_files[2])
CD3D_unregistered=np.uint8(np.array(Image.open(CD3D_unregistered)))

CD45_registered=os.path.join(registered_marker_path,registered_marker_files[3])
CD45_registered=np.uint8(np.array(Image.open(CD45_registered)))

CD45_unregistered=os.path.join(unregistered_marker_path,registered_marker_files[3])
CD45_unregistered=np.uint8(np.array(Image.open(CD45_unregistered)))

CD4_registered=os.path.join(registered_marker_path,registered_marker_files[4])
CD4_registered=np.uint8(np.array(Image.open(CD4_registered)))

CD4_unregistered=os.path.join(unregistered_marker_path,registered_marker_files[4])
CD4_unregistered=np.uint8(np.array(Image.open(CD4_unregistered)))

CD68_registered=os.path.join(registered_marker_path,registered_marker_files[5])
CD68_registered=np.uint8(np.array(Image.open(CD68_registered)))

CD68_unregistered=os.path.join(unregistered_marker_path,registered_marker_files[5])
CD68_unregistered=np.uint8(np.array(Image.open(CD68_unregistered)))

CD8_registered=os.path.join(registered_marker_path,registered_marker_files[6])
CD8_registered=np.uint8(np.array(Image.open(CD8_registered)))

CD8_unregistered=os.path.join(unregistered_marker_path,registered_marker_files[6])
CD8_unregistered=np.uint8(np.array(Image.open(CD8_unregistered)))

CGA_registered=os.path.join(registered_marker_path,registered_marker_files[7])
CGA_registered=np.uint8(np.array(Image.open(CGA_registered)))

CGA_unregistered=os.path.join(unregistered_marker_path,registered_marker_files[7])
CGA_unregistered=np.uint8(np.array(Image.open(CGA_unregistered)))

LYSOZYME_registered=os.path.join(registered_marker_path,registered_marker_files[8])
LYSOZYME_registered=np.uint8(np.array(Image.open(LYSOZYME_registered)))

LYSOZYME_unregistered=os.path.join(unregistered_marker_path,registered_marker_files[8])
LYSOZYME_unregistered=np.uint8(np.array(Image.open(LYSOZYME_unregistered)))

NAKATPASE_registered=os.path.join(registered_marker_path,registered_marker_files[9])
NAKATPASE_registered=np.uint8(np.array(Image.open(NAKATPASE_registered)))

NAKATPASE_unregistered=os.path.join(unregistered_marker_path,registered_marker_files[9])
NAKATPASE_unregistered=np.uint8(np.array(Image.open(NAKATPASE_unregistered)))

PANCK_registered=os.path.join(registered_marker_path,registered_marker_files[10])
PANCK_registered=np.uint8(np.array(Image.open(PANCK_registered)))

PANCK_unregistered=os.path.join(unregistered_marker_path,registered_marker_files[10])
PANCK_unregistered=np.uint8(np.array(Image.open(PANCK_unregistered)))

SMA_registered=os.path.join(registered_marker_path,registered_marker_files[11])
SMA_registered=np.uint8(np.array(Image.open(SMA_registered)))

SMA_unregistered=os.path.join(unregistered_marker_path,registered_marker_files[11])
SMA_unregistered=np.uint8(np.array(Image.open(SMA_unregistered)))

SOX9_registered=os.path.join(registered_marker_path,registered_marker_files[12])
SOX9_registered=np.uint8(np.array(Image.open(SOX9_registered)))

SOX9_unregistered=os.path.join(unregistered_marker_path,registered_marker_files[12])
SOX9_unregistered=np.uint8(np.array(Image.open(SOX9_unregistered)))

VIMENTIN_registered=os.path.join(registered_marker_path,registered_marker_files[13])
VIMENTIN_registered=np.uint8(np.array(Image.open(VIMENTIN_registered)))

VIMENTIN_unregistered=os.path.join(unregistered_marker_path,registered_marker_files[13])
VIMENTIN_unregistered=np.uint8(np.array(Image.open(VIMENTIN_unregistered)))

OLFM4_registered=os.path.join(registered_marker_path,registered_marker_files[14])
OLFM4_registered=np.uint8(np.array(Image.open(OLFM4_registered)))

OLFM4_unregistered=os.path.join(unregistered_marker_path,registered_marker_files[14])
OLFM4_unregistered=np.uint8(np.array(Image.open(OLFM4_unregistered)))                        
        
# List of registered and unregistered image variables
registered_images = [CD11B_registered, CD20_registered, CD3D_registered, CD45_registered, CD4_registered,
                CD68_registered, CD8_registered, CGA_registered, LYSOZYME_registered, NAKATPASE_registered,
                PANCK_registered, SMA_registered, SOX9_registered, VIMENTIN_registered, OLFM4_registered]

unregistered_images = [CD11B_unregistered, CD20_unregistered, CD3D_unregistered, CD45_unregistered, CD4_unregistered,
                CD68_unregistered, CD8_unregistered, CGA_unregistered, LYSOZYME_unregistered, NAKATPASE_unregistered,
                PANCK_unregistered, SMA_unregistered, SOX9_unregistered, VIMENTIN_unregistered, OLFM4_unregistered]

def blend_images(mask, instance, image):
    # Get the instance in the mask
    mask_instance = np.zeros_like(mask)
    mask_instance[mask == instance] = 1

    # Check if the instance exists in the mask
    if not np.any(mask_instance):
        print(f"Instance {instance} not found in the mask.")
        return None

    # Calculate the centroid of the instance
    centroid = np.mean(np.argwhere(mask_instance), axis=0)

    # Crop out a square around the centroid
    crop_size = 50
    start_x, end_x = int(centroid[0]-crop_size), int(centroid[0]+crop_size)
    start_y, end_y = int(centroid[1]-crop_size), int(centroid[1]+crop_size)

    # Boundary condition check
    start_x = max(start_x, 0)
    start_y = max(start_y, 0)
    end_x = min(end_x, image.shape[0])
    end_y = min(end_y, image.shape[1])

    mask_crop = mask[start_x:end_x, start_y:end_y]

    # Find the boundaries in the cropped mask
    boundaries = segmentation.find_boundaries(mask_crop, mode='inner')

    # Create an RGB image for boundaries
    boundary_image = np.zeros((mask_crop.shape[0], mask_crop.shape[1], 3), dtype=np.uint8)
    boundary_image[:, :, 1] = np.where((boundaries) & (mask_crop == instance), 255, 0)
    boundary_image[:, :, 2] = np.where((boundaries) & (mask_crop != instance) & (mask_crop != 0), 255, 0)

    # Create an RGB image with the cropped marker image in the red channel
    image_crop = image[start_x:end_x, start_y:end_y]
    marker_rgb = np.zeros_like(boundary_image)
    marker_rgb[:, :, 0] = image_crop
    #Make the Marker RGB image brighter by multiplying it and making sure it stays between 0,255
    marker_rgb = np.clip(marker_rgb * 30, 0, 255).astype(np.uint8)
    boundary_image = np.clip(boundary_image * 20, 0, 255).astype(np.uint8)

    # Blend the boundary image and the marker image together using numpy
    blended_image = 0.5 * boundary_image.astype(float) + 0.5 * marker_rgb.astype(float)
    blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)

    return blended_image

def display_images_with_boundaries(mask, registered_images, unregistered_images, indexes, marker_list):
    for index in indexes:
        instance_mask = mask == index
        if np.any(instance_mask):
            # Creating a figure with tightly packed subplots
            fig, axs = plt.subplots(2, len(marker_list), figsize=(20, 4), gridspec_kw={'height_ratios': [1, 1]})
            
            for i, (reg_img, unreg_img) in enumerate(zip(registered_images, unregistered_images)):
                # Blend images for the current marker
                blended_unreg = blend_images(mask, index+1, unreg_img)
                blended_reg = blend_images(mask, index+1, reg_img)
                
                # Display unregistered image in the first row
                if blended_unreg is not None:
                    axs[0, i].imshow(blended_unreg)
                    axs[0, i].axis('off')
                
                # Display registered image in the second row
                if blended_reg is not None:
                    axs[1, i].imshow(blended_reg)
                    axs[1, i].axis('off')

                # Set column titles on the first row
                    axs[0, i].set_title(marker_list[i])

            # Add annotations for row labels
            fig.text(0.01, 0.75, 'Unregistered', va='center', rotation='vertical', fontsize=12)
            fig.text(0.01, 0.25, 'Registered', va='center', rotation='vertical', fontsize=12)

            print("Registered")
            print(registered_df.loc[index])  # Assuming registered_df is predefined
            print("Unregistered")
            print(unregistered_df.loc[index])  # Assuming unregistered_df is predefined


            # Adjust the layout to make the subplots close to each other and ensure the labels are visible
            plt.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.05, hspace=0.1, wspace=0.1)
            plt.show()


# Example usage with your specific data
marker_list = ["CD11B", "CD20", "CD3D", "CD45", "CD4", "CD68", "CD8", "CGA", "LYSOZYME", "NAKATPASE", "PANCK", "SMA", "SOX9", "VIMENTIN", "OLFM4"]
display_images_with_boundaries(mask, registered_images, unregistered_images, indexes_changed, marker_list)
