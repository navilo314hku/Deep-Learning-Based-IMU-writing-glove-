import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
class DNN(nn.Module):
    def __init__(self):
        """
        layers_array: number of node for all of the hidden layer
        """
        super(DNN, self).__init__()
        self.model_name="DNN"
        self.fc0 = nn.Linear(44*6*3,560)
        self.fc1 = nn.Linear(560,256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        #print(x.shape)
        x = x.view(-1, 44*6*3)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))               # -> n, 120
        x = F.relu(self.fc2(x))               # -> n, 84
        x = self.fc3(x)  
        return x