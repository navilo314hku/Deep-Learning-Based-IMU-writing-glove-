import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
class ConvNetFlexible(nn.Module):
    def fc(self,input_node,output_node):
        return nn.Linear(input_node,output_node)
    def getTensorAmount(self,x):
        total_amount_tensor=1
        for d in range (1,x.dim()):
            total_amount_tensor*=x.size(dim=d)
        #total_amount_tensor/=self.batch_size
        total_amount_tensor=int(total_amount_tensor)
        return total_amount_tensor

    def __init__(self,output_size):
        super(ConvNetFlexible, self).__init__()
        self.model_name="ConvNetFlex"
        self.output_size=output_size
        self.conv1 = nn.Conv2d(3, 128, (10,3),padding=1,stride=(2,1))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 56, 3)
        self.pool2 = nn.MaxPool2d((2,1),(2,1))


        #self.fc1 = nn.Linear(560,256)
        #self.fc2 = nn.Linear(256, 128)
        #self.fc3 = nn.Linear(128, output_size)
        

    def forward(self, x):
        # -> n, 3, 32, 32
        x = self.pool1(F.relu(self.conv1(x)))  # -> n, 6, 14, 14
        x = self.pool2(F.relu(self.conv2(x)))  # -> n, 16, 5, 5
        x = x.view(-1, self.getTensorAmount(x))        # -> n, 400
        #print(f"dim1: {x.size(dim=1)}")#should be 560
        self.fc1=self.fc(x.size(dim=1),256)
        self.fc2=self.fc(256,128)
        self.fc3=self.fc(128,self.output_size)


        x = F.relu(self.fc1(x))               # -> n, 120
        x = F.relu(self.fc2(x))               # -> n, 84
        x = self.fc3(x)                       # -> n, 10
        return x


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