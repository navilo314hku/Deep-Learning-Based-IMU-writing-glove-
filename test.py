import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader,Dataset
from torchvision.models import resnet18

from models.model_classes.convNet import *
from models.model_classes.RNN import *
from models.customFunctions import *
from utils import *
#import torchvision.datasets.ImageFolder 
import matplotlib.pyplot as plt
import numpy as np
import os
from const import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5), (0.5))])
    
def predictSingleImage(image_path,model):
   img = Image.open(image_path)

   
   # get normalized image
   img_normalized = transform(img).float()
   #img_normalized = img_normalized.unsqueeze_(0)
   # input = Variable(image_tensor)
   img_normalized = img_normalized.to(device)
   print(img_normalized.shape)

   with torch.no_grad():
      model.eval()  
      output =model(img_normalized)
     # print(output)
      index = output.data.cpu().numpy().argmax()
      class_name = index
      return class_name

def report_accuracies(model,batch_size=batch_size,logFile=ACC_LOG_PATH):
    def report(dataloader,mode,logFile=None):
        #classes=[0,1,2,3,4,5,6,7,8,9]
        classes=[0,1,2]
        num_classes=len(classes)
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            n_class_correct = [0 for i in range(num_classes)]
            n_class_samples = [0 for i in range(num_classes)]
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                #print(outputs)
                # max returns (value ,index)
                _, predicted = torch.max(outputs, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()
                #print(f"batch_size={batch_size}")
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
                for i in range(num_classes):
                    acc = 100.0 * n_class_correct[i] / n_class_samples[i]
                    print(f'Accuracy of {classes[i]}: {acc} %')
                    f.writelines(f'Accuracy of {classes[i]}: {acc} %\n')
                    

    print("Test accuracy")
    report(test_loader,'test',ACC_LOG_PATH)
    print("Train accuracy")
    report(train_loader,'train',ACC_LOG_PATH)

if __name__=='__main__':
    #DO NOT DELETE 
    train_dataset,test_dataset,train_loader,test_loader=getDatasetDataloader()

    #DO NOT DELETE 
    input_size = 6
    hidden_size = 256
    num_layers = 2
    num_classes=3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model_path=os.path.join(Models.TRAINED_MODELS_PATH,"OptimConvNet2_20230113_151355")
    model_path="cnn.pth"
    #model=ConvNetFlexible(output_size=num_classes)
    model=resnet18()
    #model=RNN(input_size,hidden_size,num_layers,num_classes)

    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')),strict=False)
    model.eval()
    #img_path=os.path.join(realTimePrediction.ROOT_PATH,"0.jpg")
    #print(predictSingleImage(img_path,model=model))
    #model_path=os.path.join("trained_models","ConvNet2_randCrop_ep=500.pth")
    #model_path='./cnn.pth'
    report_accuracies(model)
