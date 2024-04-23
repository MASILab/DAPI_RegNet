from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

data_text_file='/fs5/p_masi/baos1/rudravg/MXIF/MXIF/Helmsley/MxIF/data_tissues.txt'
Image.MAX_IMAGE_PIXELS = None
with open(data_text_file, 'r') as file:
    lines = file.readlines()
train_files=lines[0:50]
test_files=lines[50:]

for line in tqdm(test_files):
    print(line)
    if 'TISSUE' in line:
        sub_tissue_name=(line.split('/')[4].split('_')[0])+'_'+(line.split('/')[4].split('_')[1])
        sub_name=(line.split('/')[4].split('_')[0])
    else:
        sub_name=(line.split('/')[4].split('_')[0])
        sub_tissue_name=sub_name
    set_name=line.split('/')[1]
    
    set_dir=f'/fs5/p_masi/baos1/rudravg/MXIF/MXIF/Helmsley/MxIF/{set_name}/{sub_name}/Registered'
    file_names = [name for name in os.listdir(set_dir) if sub_tissue_name in name and 'DAPI' in name]
    sorted_file_names = sorted(file_names, key=lambda name: int(name.split('_ROUND_')[1].split('.')[0]))
    target_file=f'{set_dir}/{sorted_file_names[0]}'
    source_files_1=f'{set_dir}/{sorted_file_names[10]}'
    source_files_2=f'{set_dir}/{sorted_file_names[15]}'
    source_files_3=f'{set_dir}/{sorted_file_names[-1]}'
    mask=f'/fs5/p_masi/baos1/rudravg/MXIF/MXIF/Helmsley/MxIF/{set_name}/{sub_name}/Registered/{sub_tissue_name}_RetentionMask.tif'
    target = np.array(Image.open(target_file))/255.
    source_1 = np.array(Image.open(source_files_1))/255.
    source_2 = np.array(Image.open(source_files_2))/255.
    source_3 = np.array(Image.open(source_files_3))/255.
    mask = np.array(Image.open(mask))
    target = target * mask
    source_1 = source_1 * mask
    source_2 = source_2 * mask
    source_3 = source_3 * mask
    window_size = (1024, 1024)

    # Define directories
    output_dir = '/home-local/rudravg/test_DAPI/1024_Dataset'
    target_dir = os.path.join(output_dir, 'val_target_images')
    source_dir = os.path.join(output_dir, 'val_source_images')

    # Create directories if they don't exist
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(source_dir, exist_ok=True)

    # Define pairs file
    pairs_file = os.path.join(output_dir, 'val_pairs.txt')

    # Open pairs file for writing
    with open(pairs_file, 'a') as f:
        for i in range(0, mask.shape[0], window_size[0]):
            for j in range(0, mask.shape[1], window_size[1]):
                window_mask = mask[i:i+window_size[0], j:j+window_size[1]]

                # Skip if window size is not 1024x1024
                if window_mask.shape != (1024, 1024):
                    continue

                if np.count_nonzero(window_mask) > 400:
                    # Save the cut pieces of the target in target_dir
                    window_target = target[i:i+window_size[0], j:j+window_size[1]]
                    target_path = os.path.join(target_dir, f'{sub_name}_{set_name}_Cut_{i}_{j}.npy')
                    np.save(target_path, window_target)

                    # Cut source_1, source_2 and source_3 in the same way and save their corresponding pieces in source_dir
                    for source, source_name in zip([source_1, source_2, source_3], ['source_1', 'source_2', 'source_3']):
                        window_source = source[i:i+window_size[0], j:j+window_size[1]]
                        source_path = os.path.join(source_dir, f'Sub_{sub_name}_Set_{set_name}_Round_{source_name}_Cut_{i}_{j}.npy')
                        np.save(source_path, window_source)
                        
                        # Append a space separated line in pairs_file with the first part being path to the source image and the second part being path to the target image
                        f.write(f'{source_path} {target_path}\n')