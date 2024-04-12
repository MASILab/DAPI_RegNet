from PIL import Image

import numpy as np
from scipy.ndimage import binary_dilation
# Load the image
#This script basically removes the red and green pixels that are not touching the white pixels in the image.
img = Image.open("/nfs2/baos1/rudravg/TISSUE_MASK/GCA019ACA_TISSUE_RETENTION.tif")
# New approach based on the updated instructions:
# 1. Remove all red pixels.
# 2. Keep all white pixels.
# 3. For each green pixel, check a 10x10 neighborhood for any white pixels. If a white pixel is found, keep the green pixel.
pixels = np.array(img)
white_mask_exact = (pixels[:, :, 0] == 255) & (pixels[:, :, 1] == 255) & (pixels[:, :, 2] == 255)

# Mask for red pixels (255, 0, 0)
red_mask_exact = (pixels[:, :, 0] == 255) & (pixels[:, :, 1] == 0) & (pixels[:, :, 2] == 0)

# Mask for green pixels (0, 255, 0)
green_mask_exact = (pixels[:, :, 0] == 0) & (pixels[:, :, 1] == 255) & (pixels[:, :, 2] == 0)

# Create a new mask for green pixels where we will check for the presence of white pixels in their 100x100 neighborhood
keep_green_mask = np.zeros_like(green_mask_exact)

# Iterate over the array
for x in range(50, pixels.shape[0] - 50):
    for y in range(50, pixels.shape[1] - 50):
        if green_mask_exact[x, y]:
            # Check the 100x100 neighborhood
            if np.any(white_mask_exact[x-50:x+50, y-50:y+50]):
                keep_green_mask[x, y] = True

# Now we have the mask of green pixels to keep, we remove the red pixels and the green pixels not to keep
clean_pixels_final = pixels.copy()

# Remove all red pixels
clean_pixels_final[red_mask_exact] = [0, 0, 0]

# Remove green pixels that don't have a white pixel in their neighborhood
clean_pixels_final[green_mask_exact & ~keep_green_mask] = [0, 0, 0]

binary_mask = np.zeros_like(pixels[:, :, 0])

binary_mask[white_mask_exact | keep_green_mask] = 255
binary_mask_image = np.stack((binary_mask, binary_mask, binary_mask), axis=-1)


# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# ax[0].imshow(pixels)
# ax[0].set_title('Original Image')
# ax[1].imshow(clean_pixels_final)
# ax[1].set_title('Cleaned Image')
# plt.show()

# clean_img = Image.fromarray(clean_pixels_final)
# clean_img.save('cleaned_image_neigh_10.tif')
binary_mask_img_path = "binary_mask.tif"

Image.fromarray(binary_mask_image).save(binary_mask_img_path)


binary_mask_img_path

