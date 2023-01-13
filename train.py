import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader,Dataset
from model_classes.convNet import *
from model_classes.DNN import *
from torchvision.models import resnet18
import matplotlib.pyplot as plt
from datetime import datetime
#import torchvision.datasets.ImageFolder 
import matplotlib.pyplot as plt
import numpy as np
import os
from test import report_accuracies
from const import *
from utils import *

# Device configuration
save=False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def train(model,num_epochs=500,learning_rate=0.00001):
#    MODEL_CHECKPOINT
    print(f"lr={learning_rate}")
    loss_arr=[]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    n_total_steps = len(train_loader)
    with open(LOSS_LOG_PATH, 'a') as f:
        f.writelines(f"{model.model_name}_ep{num_epochs}\n")
        f.writelines(f"training time:\n")
        f.writelines(datetime.now().strftime("%Y%m%d_%H%M%S"))
        f.write('\n')
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
            if epoch% int(num_epochs/10)==0 and save:
                print("saving model checkpoint")
                savePath=f"{model.model_name}_ep{epoch}.pth"
                savePath=os.path.join(MODEL_CHECKPOINT,savePath)
                torch.save(model.state_dict(),'./cnn.pth')
                
            f.writelines(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}\n')

        #plt.scatter(np.linspace(1, num_epochs, num_epochs).astype(int),loss_arr)
        #plt.show()
        print('Finished Training')
        f.write("Finished Training")
        training_name=model.model_name+"_"+get_datetime()
        PATH=os.path.join(TRAINED_MODELS_PATH,training_name)
        torch.save(model.state_dict(), PATH)
        torch.save(model.state_dict(), './cnn.pth')
    report_accuracies(model,batch_size=batch_size,logFile=ACC_LOG_PATH)
#Load model and train
def train_from_load(model_object,modelPath,num_epochs,learning_rate):
    
   
    #MODEL_PATH=os.path.join("trained_models","1_1_convNet2_ep=250_lr_0001.pth")
    model.load_state_dict(torch.load(modelPath,map_location=torch.device(device)),strict=False)
    model.eval()
    train(model,num_epochs=num_epochs,learning_rate=learning_rate)

def train_previous(model_object,num_epochs,learning_rate):
    model.load_state_dict(torch.load('./cnn.pth',map_location=torch.device(device)),strict=False)
    model.eval()
    train(model,num_epochs=num_epochs,learning_rate=learning_rate)

if __name__=='__main__':
    print(device)
    
    train_dataset,test_dataset,train_loader,test_loader=getDatasetDataloader()

    model=OptimConvNet2(output_size=10)
    train_previous(model,100,0.001)
    #train(model,300,0.001)
    #MODEL_PATH=os.path.join("checkpoint","ConvNetFlex_ep90.pth")
    #train_from_load(model,"cnn.pth",200,0.001)
