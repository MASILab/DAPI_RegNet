import torch
from PIL import Image
from torchvision import transforms

def mutual_information(hgram):
    """ Calculate the mutual information based on the joint histogram """
    # Convert the joint histogram to probabilities
    pxy = hgram.float() / torch.sum(hgram)
    px = torch.sum(pxy, axis=1)  # marginal for x over y
    py = torch.sum(pxy, axis=0)  # marginal for y over x
    px_py = px.unsqueeze(1) * py.unsqueeze(0)  # Broadcast to multiply marginals
    # Avoid zero values (use eps for stability)
    nzs = pxy > 0  # Non-zero singleton axes
    return torch.sum(pxy[nzs] * torch.log2(pxy[nzs] / px_py[nzs]))

def calculate_mutual_information_torch(image1, image2, bins=256):
    """ Calculate mutual information for the given images """
    # Flatten the images and convert to long type for index operations
    image1_flat = image1.reshape(-1).to(torch.int64)
    image2_flat = image2.reshape(-1).to(torch.int64)
    
    # Joint histogram
    hist_2d = torch.zeros((bins, bins), device=image1.device, dtype=torch.int64)
    indices = image1_flat * bins + image2_flat
    hist_2d.put_(indices, torch.ones_like(image1_flat, device=image1.device, dtype=torch.int64), accumulate=True)    
    # Calculate mutual information
    mi = mutual_information(hist_2d)
    
    return mi

def batch_process_mutual_information(image1_path, image2_path, bins=256, batch_size=256):
    # Ensure that PyTorch is using the GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Read the images and convert them to PyTorch tensors
    image1 = Image.open(image1_path).convert('L')
    image2 = Image.open(image2_path).convert('L')

    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [C, H, W] with range [0, 1]
    ])

    image1 = transform(image1).to(device).squeeze()  # Remove channel dim
    image2 = transform(image2).to(device).squeeze()

    # Rescale images to have values between 0 and bins-1
    image1 = (image1 * (bins-1)).int()
    image2 = (image2 * (bins-1)).int()

    # Split images into batches
    mi_values = []
    for i in range(0, image1.shape[0], batch_size):
        for j in range(0, image1.shape[1], batch_size):
            # Extract the batch from both images
            batch1 = image1[i:i+batch_size, j:j+batch_size]
            batch2 = image2[i:i+batch_size, j:j+batch_size]
            
            # Calculate mutual information for the batch
            mi_batch = calculate_mutual_information_torch(batch1, batch2, bins)
            mi_values.append(mi_batch)

    # Combine MI values from all batches
    total_mi = torch.stack(mi_values).mean().item()

    return total_mi # Convert to Python scalar
image1_path = '/nfs2/baos1/rudravg/GCA007TIB_TISSUE01_DAPI_DAPI_12ms_ROUND_18_initial_reg.tif'
image2_path = '/nfs2/baos1/rudravg/GCA007TIB_TISSUE01_DAPI_DAPI_12ms_ROUND_18_initial_reg.tif'
#result_mi = calculate_mutual_information(image1_path, image2_path)
result_mi = batch_process_mutual_information(image1_path, image2_path)
print(result_mi)