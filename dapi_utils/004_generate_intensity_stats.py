import numpy as np
import pandas as pd
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import os
import re
Image.MAX_IMAGE_PIXELS = None

marker_list = ['CD11B','CD20','CD3d','CD45','CD4','CD68','CD8','CgA','Lysozyme','NaKATPase','PanCK','SMA','Sox9','Vimentin','OLFM4']

image_dir='/fs5/p_masi/rudravg/MxIF_Vxm_Registered_V2/GCA020TIB_TISSUE01/Unregistered/AF_Removed'

marker_files = {}

# Loop over each marker
for marker in marker_list:
    # Search for files that contain the marker name
    files = glob.glob(os.path.join(image_dir, f"*{marker}*"))
    
    # Filter the files to only include those that exactly match the marker name
    files = [file for file in files if f"GCA020TIB_TISSUE01_{marker}_" in file.split('/')[-1]]    
    # Add the files to the dictionary
    marker_files[marker] = files

print(marker_files)

data=[]

mask = Image.open('/fs5/p_masi/rudravg/MxIF_Vxm_Registered_V2/GCA020TIB_TISSUE01/mask.tif')
print(f'Instance Mask path is /fs5/p_masi/rudravg/MxIF_Vxm_Registered_V2/GCA020TIB_TISSUE01/mask.tif')
mask_np = np.array(mask)
unique_instances = np.unique(mask_np)
unique_instances = unique_instances[unique_instances != 0] 

import concurrent.futures
from tqdm import tqdm

def process_instance(instance, preloaded_images):
    # Initialize a list to hold the current row of data
    row = [instance]

    # Create a mask for the current instance
    instance_mask = mask_np == instance

    # Calculate the centroid of the current instance
    y_indices, x_indices = np.where(instance_mask)
    centroid_x = np.mean(x_indices)
    centroid_y = np.mean(y_indices)

    # Append the centroid to the row
    row.extend([centroid_x, centroid_y])

    # Loop over each marker
    for marker, image_np in preloaded_images.items():
        # Get the pixels of the current instance
        instance_pixels = image_np[instance_mask]

        # Calculate the mean intensity
        mean_intensity = np.mean(instance_pixels)

        # Append the mean intensity to the row
        row.append(mean_intensity)

    return row

# Pre-load all images into memory
preloaded_images = {}
for marker, files in marker_files.items():
    file = files[0]  # Assume there's only one file per marker
    image = Image.open(file)
    preloaded_images[marker] = np.array(image)

# Initialize a list to hold the data
data = []

# Create a ThreadPoolExecutor
with concurrent.futures.ThreadPoolExecutor() as executor:
    # Use the executor to map the process_instance function to the unique_instances
    # Pass preloaded_images to each fnction call
    tasks = [executor.submit(process_instance, instance, preloaded_images) for instance in unique_instances]
    for future in tqdm(concurrent.futures.as_completed(tasks), total=len(unique_instances)):
        data.append(future.result())
marker_list = ['Mean_' + marker for marker in marker_list]

df1 = pd.DataFrame(data, columns=['Instance', 'Centroid_X', 'Centroid_Y'] + marker_list)
df1=df1.sort_values('Instance')
df1['slide_id'] = 1

df1.to_csv('/fs5/p_masi/rudravg/MxIF_Vxm_Registered_V2/GCA020TIB_TISSUE01/unregistered_GCA020TIB_instances.csv', index=False)
