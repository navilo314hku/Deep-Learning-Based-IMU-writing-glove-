from const import *
from datetime import datetime
from matplotlib import image
import numpy as np 
import cv2
import os 
from time import sleep
from customFunctions import *
def removeTxt():
    import os
    txt_dir = os.path.join("/Users/ivanlo/Desktop/FYP/code/main/",TXT_PATH)
    txtFileList = os.listdir(txt_dir)
    for item in txtFileList:
        if item.endswith(".txt"):
            os.remove(os.path.join(txt_dir, item))
def get_datetime():
    return datetime.now().strftime("%Y%m%d_%H%M%S")
def report_data():
    train_folders=os.listdir(TRAIN_IMAGE_PATH)
    train_folders.sort()
    paths=[TRAIN_IMAGE_PATH,TEST_IMAGE_PATH]
    for path in paths:#paths=[...train, .../test]
        print(path)
        folder_list=os.listdir(path)
        folder_list.sort()
        for folder in folder_list:#
            if os.path.isdir(os.path.join(path,folder)):
                folder_dir=os.path.join(path,folder)
                num_of_files=len(os.listdir(folder_dir))
                print(f"{folder}: {num_of_files}")
def getDatasetDataloader():
    train_dataset=torchvision.datasets.ImageFolder(TRAIN_IMAGE_PATH,transform=randCropTransform,loader=custom_pil_loader)
    test_dataset= torchvision.datasets.ImageFolder(TEST_IMAGE_PATH,transform=basicTransform,loader=custom_pil_loader)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=True)
    return train_dataset,test_dataset,train_loader,test_loader

def showImg0_255(img_dir):
    print(TEST_IMAGE_PATH)
    path=os.path.join(TEST_IMAGE_PATH,"1","20230102_014555.jpg")
    print(path) 
    arr=image.imread(path)
    print(type(arr))
    new_arr = ((arr - arr.min()) * (1/(arr.max() - arr.min()) * 255)).astype('uint8')
    cv2.imwrite("1.jpg",new_arr)
    
if __name__=="__main__":
    #report_data()
    showImg0_255("")