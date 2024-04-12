import torch
import torchvision.transforms as transforms
from PIL import Image

def calculate_normalized_cross_correlation_torch(image1_path, image2_path):
    # Ensure that PyTorch is using the GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Read the images and convert them to PyTorch tensors
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [C, H, W] with range [0, 1]
    ])

    image1 = transform(image1).unsqueeze(0).to(device)  # Adds a batch dimension [B, C, H, W]
    image2 = transform(image2).unsqueeze(0).to(device)

    # Normalize the images
    image1 = (image1 - torch.mean(image1)) / torch.std(image1)
    image2 = (image2 - torch.mean(image2)) / torch.std(image2)

    # Flatten the images to 1D vectors
    image1_flat = image1.view(-1)
    image2_flat = image2.view(-1)

    # Calculate the normalized cross-correlation coefficient as a single value
    ncc_coefficient = (torch.dot(image1_flat, image2_flat) /
                       (torch.norm(image1_flat) * torch.norm(image2_flat))).item()  # Convert to Python scalar

    return ncc_coefficient



# Example usage
image1_path = '/nfs2/baos1/rudravg/GCA007TIB_TISSUE01_DAPI_DAPI_12ms_ROUND_18_initial_reg.tif'
image2_path = '/nfs2/baos1/rudravg/GCA007TIB_TISSUE01_DAPI_DAPI_12ms_ROUND_18_initial_reg.tif'
#result_mi = calculate_mutual_information(image1_path, image2_path)
result_ncc = calculate_normalized_cross_correlation_torch(image1_path, image2_path)
print(result_ncc)
