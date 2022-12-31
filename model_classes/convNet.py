import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
class ConvNet2(nn.Module):
    
    def __init__(self,output_size):
        super(ConvNet2, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3,padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 56, 3)
        self.pool2 = nn.MaxPool2d((2,1),(2,1))


        self.fc1 = nn.Linear(560,256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        # -> n, 3, 32, 32
        x = self.pool1(F.relu(self.conv1(x)))  # -> n, 6, 14, 14
        x = self.pool2(F.relu(self.conv2(x)))  # -> n, 16, 5, 5
        x = x.view(-1, 560)            # -> n, 400
        x = F.relu(self.fc1(x))               # -> n, 120
        x = F.relu(self.fc2(x))               # -> n, 84
        x = self.fc3(x)                       # -> n, 10
        return x

class ConvNet(nn.Module):
    def __init__(self,output_size):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 2)
        self.fc1 = nn.Linear(20*1*16, 120)
        self.fc2 = nn.Linear(120, 50)
        self.fc3 = nn.Linear(50, output_size)

    def forward(self, x):
        # -> n, 3, 32, 32
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 14, 14
        x = F.relu(self.conv2(x))  # -> n, 16, 5, 5
        x = x.view(-1, 20*1*16)            # -> n, 400
        x = F.relu(self.fc1(x))               # -> n, 120
        x = F.relu(self.fc2(x))               # -> n, 84
        x = self.fc3(x)                       # -> n, 10
        return x