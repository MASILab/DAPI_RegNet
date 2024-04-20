from PIL import Image
import os

def upsample_tiff(image_path, output_path):
    # Open the TIFF image
    with Image.open(image_path) as img:
        # Calculate new size (double the original dimensions)
        new_size = tuple(2 * x for x in img.size)
        
        # Resize the image using bicubic interpolation
        upscaled_img = img.resize(new_size, Image.BICUBIC)
        
        # Save the upscaled image
        upscaled_img.save(output_path, format='TIFF')

def process_directory(input_dir, output_dir):
    # Check if output directory exists, create if not
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Iterate over all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.tiff') or filename.lower().endswith('.tif'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            print(f"Processing {filename}...")
            upsample_tiff(input_path, output_path)
            print(f"Saved upscaled image to {output_path}")

# Example usage
input_directory = '/nfs2/baos1/rudravg/deep_cell_original_resolution/downsampled_resolution'
output_directory = '/nfs2/baos1/rudravg/deep_cell_original_resolution/original_resolution'
deep_cell_path='/nfs2/baos1/rudravg/deep_cell_original_resolution/original_resolution/deep_cell_results'
process_directory(input_directory, output_directory)
