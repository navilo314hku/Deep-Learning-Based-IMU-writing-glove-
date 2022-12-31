24/12/2022
Store txt files from txtStorage directory into jpg files in 
Images/{label}/{date_time}.jpg images


28/12
trained ConvNet model with around 100 data per number
tested 10 data per number
train acc:95%
test accuracy: 58%

29/12 
Tried importing resnet18 model from pytorch library 
Tried HKU gpu, still valid 
test.py: add accuracy for each labels.
29_12.pth
Accuracy of the network: 63.41463414634146 %
Accuracy of 0: 87.5 %
Accuracy of 1: 62.5 %
Accuracy of 2: 62.5 %
Accuracy of 3: 14.285714285714286 %
Accuracy of 4: 62.5 %
Accuracy of 5: 25.0 %
Accuracy of 6: 100.0 %
Accuracy of 7: 88.88888888888889 %
Accuracy of 8: 75.0 %
Accuracy of 9: 50.0 %

TODO: add 250 more data to '3' and '5' 

31/12
Implement convNet2 which has following architecture
Input (3*44*6)
conv1: filter size:3*3 num filter=128 padding=1
maxpooling (2*2)
conv2: filter size: 3*3 num filter=56 
maxpolling (2*1)
fully connected layer 560=>256>128>10