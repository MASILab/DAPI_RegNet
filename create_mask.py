from PIL import Image
import numpy as np
import os
from tqdm import tqdm
data_dir = "/nfs2/baos1/rudravg/TISSUE_MASK/"
save_dir = "/nfs2/baos1/rudravg/Retention_Masks/"
Image.MAX_IMAGE_PIXELS = None

def process_image(filename):
    img = Image.open(data_dir + filename)
    img = img.resize(((img.width // 2)+1, (img.height // 2)+1))
    pixels = np.array(img)
    num_white = 0
    middle_column_index = pixels.shape[1] // 2

# Iterate over pixels in the middle column
    for pixel in pixels[:, middle_column_index]:
        if np.all(pixel == 0):
            break
        else:
            num_white += 1
    white_mask_exact = (pixels[:, :, 0] == 255) & (pixels[:, :, 1] == 255) & (pixels[:, :, 2] == 255)
    white_mask = white_mask_exact[num_white-1:, :]
    binary_mask_img_path = save_dir + filename
    binary_mask_img = Image.fromarray(white_mask.astype(np.uint8) * 255)
    binary_mask_img.save(binary_mask_img_path)

# Load every image in the data_dir and save the binary mask in the save_dir
file_list = os.listdir(data_dir)
for filename in tqdm(file_list):
    print(filename)
    process_image(filename)
"""
# Load the image
img = Image.open("/nfs2/baos1/rudravg/TISSUE_MASK/GCA019ACA_TISSUE_RETENTION.tif")
print(img.size)

#Downsample the image by a factor of 2
img = img.resize(((img.width // 2)+1, (img.height // 2)+1))
half_width=img.size[0]//2
#For the column denoted by half_width, find the first black pixel from the top

# Convert image to numpy array
pixels = np.array(img)
num_white=0

for pixel in pixels:
    if np.all(pixel==0):
        break
    else:
        num_white+=1
print(num_white)

# Create a binary mask for white pixels
white_mask = (pixels[:, :, 0] == 255) & (pixels[:, :, 1] == 255) & (pixels[:, :, 2] == 255)
white_mask=white_mask[num_white:,:]



# Display the binary mask
import matplotlib.pyplot as plt
plt.imshow(white_mask, cmap='gray')
plt.title('Binary Mask')
plt.show()

#Remove the first num_white rows from the binary mask

# Save the binary mask as an image
binary_mask_img = Image.fromarray(white_mask.astype(np.uint8) * 255)
binary_mask_img.save('binary_mask.tif')
"""