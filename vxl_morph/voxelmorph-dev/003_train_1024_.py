import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'

import voxelmorph as vxm
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor

class ImageRegistrationDataset(Dataset):
    def __init__(self, pairs_file):
        self.image_pairs = []
        with open(pairs_file, 'r') as file:
            for line in file:
                moving, fixed = line.strip().split()
                self.image_pairs.append((moving, fixed))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        moving_path, fixed_path = self.image_pairs[idx]

        # Load images as numpy arrays
        moving_image = np.load(moving_path)
        fixed_image = np.load(fixed_path)

        # Convert numpy arrays to PyTorch tensors and add the batch dimension
        # Also, permute dimensions to match the desired shape: [batch_size, channels, height, width]
        moving_tensor = torch.from_numpy(moving_image).unsqueeze(0).float()
        fixed_tensor = torch.from_numpy(fixed_image).unsqueeze(0).float()

        return [moving_tensor, fixed_tensor], [fixed_tensor]


##Create the dataset and dataloader
train_dataset = ImageRegistrationDataset('/home-local/rudravg/test_DAPI/1024_Dataset/train_pairs.txt')
val_dataset = ImageRegistrationDataset('/home-local/rudravg/test_DAPI/1024_Dataset/val_pairs.txt')
train_data_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=5, shuffle=True)

# Define the model
IN_SHAPE=(1024,1024)
model=vxm.networks.VxmDense(inshape=IN_SHAPE)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
model.to(device)
model.train()
optimizer=torch.optim.Adam(model.parameters(), lr=1e-3)

# Define the loss function
image_loss_func=vxm.losses.NCC().loss 
losses=[image_loss_func]
weights=[1]

losses += [vxm.losses.Grad('l2',loss_mult=2).loss]
weights += [0.01]


epoch_loss_final=[]
epoch_val_loss_final=[]
inference_dir='/home-local/rudravg/test_DAPI/1024_Dataset/epochs/'
best_val_loss = float('inf')
patience = 50
epochs_no_improve = 0


##Training the model
for epoch in range(500):
    ########### Training ###########
    model.train()
    epoch_loss=[]
    epoch_total_loss=[]
    val_loss=[]

    for inputs, y_true in train_data_loader:
        inputs = [i.to(device) for i in inputs]
        y_true = [i.to(device) for i in y_true]
        y_pred=model(*inputs)
        loss=0
        loss_list=[]
        for n,loss_function in enumerate(losses):
            curr_loss=loss_function(y_true[0],y_pred[0])*weights[n]
            loss_list.append(curr_loss.item())
            loss+=curr_loss
        epoch_loss.append(loss_list)
        epoch_total_loss.append(loss.item())    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    epoch_loss_final.append(np.mean(epoch_total_loss))

    ########### Inference ###########
    with torch.inference_mode():
        epoch_dir = os.path.join(inference_dir, f'epoch_{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)
        model.save(os.path.join(epoch_dir, f'epoch_{epoch}.pth'))
        model.eval()
        i=0
        for inputs, y_true in val_data_loader:
            i+=1
            inputs = [i.to(device) for i in inputs]
            y_true = [i.to(device) for i in y_true]
            y_pred=model(*inputs)
            loss=0
            for n,loss_function in enumerate(losses):
                curr_loss=loss_function(y_true[0],y_pred[0])*weights[n]
                loss+=curr_loss
            val_loss.append(loss.item())
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(inputs[0][0].cpu().numpy().squeeze(), cmap='gray')
            plt.title('Moving Image')
            plt.subplot(1, 3, 2)
            plt.imshow(inputs[1][0].cpu().numpy().squeeze(), cmap='gray')
            plt.title('Fixed Image')
            plt.subplot(1, 3, 3)
            plt.imshow(y_pred[0][0].detach().cpu().numpy().squeeze(), cmap='gray')
            plt.title('Registered Image')
            plt.savefig(os.path.join(epoch_dir, f'pair_{i}.png'))  # Save the plot as an image
            plt.close()
        epoch_val_loss_final.append(np.mean(val_loss))
    mean_val_loss = np.mean(val_loss)
    if mean_val_loss < best_val_loss:
        best_val_loss = mean_val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
    if epochs_no_improve == patience:
        print(f'Early stopping on epoch {epoch}')
        break
    print('------------------------------------------------------')
    print(f'Epoch: {epoch} Loss: {np.mean(epoch_total_loss)}')
    print(f'Epoch {epoch} Val Loss: {np.mean(val_loss)}')
    print(f'Epoch {epoch} Losses: {np.mean(epoch_loss, axis=0)}')
    print('------------------------------------------------------')
