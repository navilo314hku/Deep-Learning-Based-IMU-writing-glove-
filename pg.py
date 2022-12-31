import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader,Dataset
from model_classes.convNet import ConvNet,ConvNet2
from torchvision.models import resnet18
import matplotlib.pyplot as plt

#import torchvision.datasets.ImageFolder 
import matplotlib.pyplot as plt
import numpy as np
import os
from const import *
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size=4
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])

import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        #print(f"image path: {img_path}")
        image = read_image(img_path)
        image=image.to(torch.float32)
        #print(f"image type: {type(image)}")
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            #print('here')
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
train_dataset= torchvision.datasets.ImageFolder(TRAIN_IMAGE_PATH,transform)
test_dataset= torchvision.datasets.ImageFolder(TEST_IMAGE_PATH,transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=True)
def train(model,num_epochs=500,learning_rate=0.001):
    print(f"lr={learning_rate}")
    loss_arr=[]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # origin shape: [4, 3, 32, 32] = 4, 3, 1024
            # input_layer: 3 input channels, 6 output channels, 5 kernel size
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_arr.append(loss)
            if (i+1) % 200 == 0:
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
    #plt.scatter(np.linspace(1, num_epochs, num_epochs).astype(int),loss_arr)
    #plt.show()
    print('Finished Training')
    PATH = './cnn.pth'
    torch.save(model.state_dict(), PATH)

#Train from scratch
output_size=10
print(device)
model = ConvNet2(output_size=output_size).to(device)
#model= resnet18().to(device)
model.eval()
train(model)

