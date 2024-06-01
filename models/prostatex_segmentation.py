# -*- coding: utf-8 -*-
"""ProstateX_Segmentation

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/186RBni34J_3To7Zr-6EJt0v9hq0Mfusk
"""

import matplotlib.pyplot as plt
import torch
import numpy as np
from monai.losses import DiceCELoss, DiceLoss
from torch.utils.data import DataLoader, random_split

from prostatexdataset import ProstateXDataset
from unet import UNet

torch.manual_seed(0)

# set appropriate device
mode = "debug"
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using CUDA")
    mode = "run"
else:
    device = torch.device('cpu')
    print("Using CPU")

# initialize dataset and model
dataset = ProstateXDataset(root_dir='./Data/PROSTATEx')
model = UNet(1, 3).to(device)

# set up parameters
if mode == "debug":
    batch_size = 1
    num_epochs = 1
elif mode == "run":
    batch_size = 5
    num_epochs = 2

lr = 4.3e-3
beta1 = 0.9
beta2 = 0.95 #0.999
weight_decay = 0.05 #1e-5

optimizer = torch.optim.AdamW(model.parameters(),
                                            lr=lr,
                                            betas=(beta1, beta2),
                                            weight_decay=weight_decay)

smooth_nr = 0.0
smooth_dr = 1e-6

criterion = DiceCELoss(softmax=True,
                                    squared_pred=True,
                                    smooth_nr=smooth_nr,
                                    smooth_dr=smooth_dr)


# perform train-test split
dataset_size = len(dataset)
if mode == "debug":
    train_size = int(0.5 * dataset_size)
elif mode == "run":
    train_size = int(0.8 * dataset_size)


train_dataset, test_dataset = random_split(dataset, [train_size, dataset_size - train_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Total number of training scans:", len(train_loader.dataset))

# training loop
print("Training...")
losses = np.array([])
for epoch in range(num_epochs):
  for batch_idx, (inputs, targets) in enumerate(train_loader):
    inputs, targets = inputs.to(device).float(), targets.to(device)
    inputs = inputs[None, :, :, :, :] # use 1 input in batch but pretend it's a whole batch
    optimizer.zero_grad()

    outputs = model(inputs)
    loss = criterion(outputs.permute((0,1,3,4,2)), targets.permute((0,1,3,4,2)))
    losses = np.append(losses, loss.item())

    loss.backward()
    optimizer.step()
    print(f"Batch {batch_idx+1}, Loss: {loss.item()}")


# plot training performance
plt.figure(figsize=(10, 6))
plt.plot(losses, marker='o')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.title('DiceCELoss Curve with Improved Params')
plt.grid(True)
plt.show()

# run test data
print("Testing...")
loss = 0
for batch_idx, (inputs, targets) in enumerate(test_loader):
  inputs, targets = inputs.to(device).float(), targets.to(device)
  inputs = inputs[None, :, :, :, :] # use 1 input in batch but pretend it's a whole batch

  outputs = model(inputs)
  loss += criterion(outputs.permute((0,1,3,4,2)), targets.permute((0,1,3,4,2)))
  print(f"Batch {batch_idx+1}, Loss: {loss.item()}")