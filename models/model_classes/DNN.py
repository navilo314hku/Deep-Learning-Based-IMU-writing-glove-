import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
class DNN_base(nn.Module):
    def __init__(self,l):#l=[120,50,25,10]
        super(DNN_base,self).__init__()
        print("DNN_base init start ")
        self.model_name="DNN_base"
        self.l=l
        self.fc0= nn.Linear(l[0],l[1])
        self.fc1 = nn.Linear(l[1],l[2])
        self.fc2 = nn.Linear(l[2],l[3])
        self.fc3 = nn.Linear(l[3],l[4])
        
    def forward(self,x):
        x = x.view(-1,self.l[0])
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))               # -> n, 120
        x = F.relu(self.fc2(x))               # -> n, 84
        x = self.fc3(x) 
class DNN_child(DNN_base):
    def __init__(self):
        self.parentModel=super(DNN_child,self)
        self.parentModel.__init__(l=[44*6,560,256,128,10])
        print("DNN_child init complete")
        #self.parentModel.__init__(l=[44*6*3,560,256,128,10])
    def forward(self,x):
        print(x.size())
        self.parentModel.forward(x)
        return x
    


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