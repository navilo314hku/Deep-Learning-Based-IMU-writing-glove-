import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader,Dataset
from torch.utils.tensorboard import SummaryWriter

from torchvision.models import resnet18
import matplotlib.pyplot as plt
from datetime import datetime
#import torchvision.datasets.ImageFolder 
import numpy as np
import os
#custom import
from models.model_classes.convNet import *
from models.model_classes.DNN import *
from models.model_classes.RNN import *
from const import *
from test import report_accuracies
from utils import *


# Device configuration
save=0
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

def train(model,writer_name="",num_epochs=500,learning_rate=0.00001):
#    MODEL_CHECKPOINT
    writer=SummaryWriter(f"runs/{writer_name}")
    print(f"lr={learning_rate}")
    print(f"batch size={batch_size}")
    loss_arr=[]
    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    n_total_steps = len(train_loader)
    with open(LOSS_LOG_PATH, 'a') as f:
        #f.writelines(f"{model.model_name}_ep{num_epochs}\n")
        f.writelines(f"training time:\n")
        f.writelines(datetime.now().strftime("%Y%m%d_%H%M%S"))
        f.write('\n')

        print(len(train_loader))
        for epoch in range(num_epochs):
            running_loss=0.0
            running_correct=0
            for i, (images, labels) in enumerate(train_loader):#loop for one batch
                #print(f"length of label={len(labels)}")
                #print(len(labels))
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
                running_loss+=loss.item()
                _, predicted = torch.max(outputs.data, 1)
                running_correct+=(predicted==labels).sum().item()
                
                
                if (i+1) % 200 == 0:# print loss
                    print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            train_acc,test_acc=report_accuracies(model,train_loader,test_loader,print_result=0)
            print("writing tensorboard")
            writer.add_scalar("training accuracy",train_acc,epoch)
            writer.add_scalar("testing accuracy",test_acc,epoch)
        print('Finished Training')
        f.write("Finished Training")
        training_name=model.model_name+"_"+get_datetime()
        PATH=os.path.join(TRAINED_MODELS_PATH,training_name)
        #torch.save(model.state_dict(), PATH)
        torch.save(model.state_dict(), './cnn.pth')
    report_accuracies(model,train_loader,test_loader,batch_size=batch_size,logFile=ACC_LOG_PATH)
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
    # input_size = 6
    # hidden_size = 256
    # num_layers = 2
    #sequence_length = 44
    train_dataset,test_dataset,train_loader,test_loader=getDatasetDataloader()
    num_classes = len(train_dataset.classes)
    print(f"number of classes: {num_classes}")
    
    #model=ConvNetFlexible(output_size=num_classes)
    model=OptimConvNet2(output_size=num_classes)
    print(f"using model: {model.model_name}")
    #model=DNN()
    #model=RNN(input_size,hidden_size,num_layers,num_classes)
    #model = RNN_LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
    #model=resnet18()
    #model = RNN2(n_input=input_size,n_output=num_classes,n_hidden=20)
    #train(model,200,0.0005)
    #train(model,100,0.000125)
    #MODEL_PATH=os.path.join("checkpoint","ConvNetFlex_ep90.pth")
    print(model.eval())
    lrs=[0.001,0.0005,0.00025,0.000125]
    for lr in lrs:
        train(model,writer_name=str(lr),num_epochs=4,learning_rate=lr)
        #train_from_load(model,"cnn.pth",50,0.0005)
    #train(model,100,0.001)
