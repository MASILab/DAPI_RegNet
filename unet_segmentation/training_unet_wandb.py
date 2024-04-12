import wandb
import random
wandb.login()

from unet_model import UNet
import torch
from torch.utils.data import Dataset,DataLoader,random_split
from pathlib import Path
import numpy as np
from PIL import Image
from torch import nn
from tqdm import tqdm
import torch.optim as optim

###############################Dataloader##################

class Segmentation_Dataloader(Dataset):
    def __init__(self, images_dir: str, masks_dir: str):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.ids = [file.stem for file in self.images_dir.glob('*') if file.is_file()]
        self.ids.sort()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_name = self.ids[idx]

        img_path = self.images_dir / f'{img_name}.npy'
        mask_path = self.masks_dir / f'{img_name}_mask.npy'

        assert img_path.exists(), f"Image file {img_path} does not exist"
        assert mask_path.exists(), f"Mask file {mask_path} does not exist"

        img = np.load(img_path)
        mask = np.load(mask_path)

        assert img.shape[:2] == mask.shape[:2], f"Image and mask shapes do not match for {img_name}"

        # Resize image and mask
        img = np.moveaxis(img, -1, 0)  # Move channel axis to front
        mask = np.moveaxis(mask, -1, 0)  # Move channel axis to front

        img = torch.tensor(img, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        img = img.squeeze(0)
        mask = mask.squeeze(0)

        return {'image': img.unsqueeze(0), 'mask': mask.unsqueeze(0),'mask_path':str(mask_path)}
    
####Loss Function####
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        smooth = 0.00001
        input_flat = input.view(-1)
        target_flat = target.view(-1)
        intersection = (input_flat * target_flat).sum()
        dice_coeff = (2. * intersection + smooth) / (input_flat.sum() + target_flat.sum() + smooth)
        return 1 - dice_coeff
    

############Setting up the Dataset and Model####################
dataset=Segmentation_Dataloader(images_dir='/home-local/rudravg/Segmentation_test/Images', masks_dir='/home-local/rudravg/Segmentation_test/Masks')
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

train_size = int(0.6 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

model=UNet(n_channels=1,n_classes=1)

criterion = DiceLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

###################Training Loop####################
def train(model, criterion, optimizer, train_loader, device):
    model.train()
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        inputs, targets = data['image'].to(device), data['mask'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        probs = torch.sigmoid(outputs)
        loss = criterion(probs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def validate(model, criterion, val_loader, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in val_loader:
            inputs, targets = data['image'].to(device), data['mask'].to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            val_loss += criterion(probs, targets).item()
    return val_loss / len(val_loader)

def test(model, criterion, test_loader, device):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for data in test_loader:
            inputs, targets = data['image'].to(device), data['mask'].to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            test_loss += criterion(probs, targets).item()
    return test_loss / len(test_loader)
########Intitalising Wandb################
run = wandb.init(

    project="Unet_Segmentation",

    config={
        "learning_rate": 0.001,
        "epochs": 10,
        "Loss fn": "Dice Loss",
        "Batch Size":1
    },
)

num_epochs = 10
device='cuda' if torch.cuda.is_available() else 'cpu'
model=model.to(device)
best_val_loss = float('inf')
for epoch in tqdm(range(num_epochs)):
    train_loss = train(model, criterion, optimizer, train_loader, device)
    val_loss = validate(model, criterion, val_loader, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    wandb.log({"Train Loss": train_loss, "Val Loss": val_loss})
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')    
wandb.finish()