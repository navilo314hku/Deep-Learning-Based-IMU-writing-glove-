import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader,Dataset
from model_classes.convNet import ConvNet,ConvNet2
#import torchvision.datasets.ImageFolder 
import matplotlib.pyplot as plt
import numpy as np
import os
from const import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])

def report_accuracies(model,batch_size=4,logFile=ACC_LOG_PATH):
    def report(dataloader,mode,logFile=None):
        classes=[0,1,2,3,4,5,6,7,8,9]
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            n_class_correct = [0 for i in range(10)]
            n_class_samples = [0 for i in range(10)]
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                #print(outputs)
                # max returns (value ,index)
                _, predicted = torch.max(outputs, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

                for i in range(batch_size):
                    contin=0
                    for j in range(1,batch_size):
                        if(labels.size()==torch.Size([j])):

                            contin=1
                    if contin:
                        continue
                    label = labels[i]
                    pred = predicted[i]
                    if (label == pred):
                        n_class_correct[label] += 1
                    n_class_samples[label] += 1

            acc = 100.0 * n_correct / n_samples
            with open(logFile,'a') as f:
                if mode=='test':
                    print(f'Test Accuracy of the network: {acc} %')
                    f.writelines(f'Test Accuracy of the network: {acc} %\n')
                elif mode=='train':
                    print(f'Train Accuracy of the network: {acc} %')
                    f.writelines(f'Train Accuracy of the network: {acc} %\n')
                else: 

                    raise Exception('Invalid mode')
                for i in range(10):
                    acc = 100.0 * n_class_correct[i] / n_class_samples[i]
                    print(f'Accuracy of {classes[i]}: {acc} %')
                    f.writelines(f'Accuracy of {classes[i]}: {acc} %\n')
                    
    test_dataset= torchvision.datasets.ImageFolder(TEST_IMAGE_PATH,transform)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=True)
    train_dataset = torchvision.datasets.ImageFolder(TRAIN_IMAGE_PATH,transform)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
    print("Test accuracy")
    report(testloader,'test',ACC_LOG_PATH)
    print("Train accuracy")
    report(trainloader,'train',ACC_LOG_PATH)

if __name__=='__main__':
    model = ConvNet2(output_size=10)
    MODEL_PATH=os.path.join("trained_models","1_1_convNet2_ep=250_lr_0001.pth")
    #MODEL_PATH="cnn.pth"
    model.load_state_dict(torch.load(MODEL_PATH,map_location=torch.device('cpu')))
    model.eval()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))])

    report_accuracies(model)
