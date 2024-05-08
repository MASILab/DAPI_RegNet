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

marker_list = ['CD11B','CD20','CD3D','CD45','CD4','CD68','CD8','CGA','LYSOZYME','NAKATPASE','PANCK','SMA','SOX9','VIMENTIN','OLFM4']

image_dir='/fs5/p_masi/rudravg/MxIF_Vxm_Registered/GCA012TIB/'

marker_files = {}

# Loop over each marker
for marker in marker_list:
    # Search for files that contain the marker name
    files = glob.glob(os.path.join(image_dir, f"*{marker}*"))
    
    # Filter the files to only include those that exactly match the marker name
    files = [file for file in files if file.split('/')[-1].startswith(f"GCA012TIB_{marker}_")]
    
    # Add the files to the dictionary
    marker_files[marker] = files

data=[]

mask = Image.open('/fs5/p_masi/rudravg/MxIF_Vxm_Registered/GCA012TIB/mask.tif')
mask_np = np.array(mask)
unique_instances = np.unique(mask_np)
unique_instances = unique_instances[unique_instances != 0] 

import concurrent.futures
from tqdm import tqdm

def process_instances(instances, marker_files, mask_np):
    # Initialize a list to hold all rows of data
    rows = []

    # Loop over each marker
    for marker, files in marker_files.items():
        # Assume there's only one file per marker
        file = files[0]

        # Open the image file
        image = Image.open(file)

        # Convert the image to a numpy array
        image_np = np.array(image)

        # Loop over each instance
        for instance in instances:
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

            # Get the pixels of the current instance
            instance_pixels = image_np[instance_mask]

            # Calculate the mean intensity
            mean_intensity = np.mean(instance_pixels)

            # Append the mean intensity to the row
            row.append(mean_intensity)

            # Append the row to the rows
            rows.append(row)

    return rows

# Initialize a list to hold the data
from tqdm import tqdm

# Initialize a list to hold the data
data = []

# Create a ThreadPoolExecutor
with concurrent.futures.ThreadPoolExecutor() as executor:
    # Use the executor to map the process_instances function to the unique_instances
    future_results = {executor.submit(process_instances, instance, marker_files, mask_np): instance for instance in unique_instances}
    for future in tqdm(concurrent.futures.as_completed(future_results), total=len(future_results)):
        data.extend(future.result())

df1 = pd.DataFrame(data, columns=['Instance', 'Centroid_X', 'Centroid_Y'] + marker_list)

df1.to_csv('/fs5/p_masi/rudravg/MxIF_Vxm_Registered/GCA012TIB/registered_GCA012TIB_instances.csv', index=False)