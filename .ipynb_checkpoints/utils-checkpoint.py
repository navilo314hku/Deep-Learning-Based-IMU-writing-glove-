from const import *
from datetime import datetime
import numpy as np 
import cv2
import os 
from time import sleep

def removeTxt():
    import os
    txt_dir = os.path.join("/Users/ivanlo/Desktop/FYP/code/main/",TXT_PATH)
    txtFileList = os.listdir(txt_dir)
    for item in txtFileList:
        if item.endswith(".txt"):
            os.remove(os.path.join(txt_dir, item))

def report_data():
    #output: 
    """
    train:
    0: amount
    1: amount
    2: ...
    .
    .
    .
    9
    
    test:
    0: amount
    1: 
    .
    .
    .
    9
    
    
    """
    #TRAIN_IMAGE_PATH=os.path.join("images","train")
    #TEST_IMAGE_PATH=os.path.join("images","test")
    #list 
    #train image
    train_folders=os.listdir(TRAIN_IMAGE_PATH)
    train_folders.sort()
    paths=[TRAIN_IMAGE_PATH,TEST_IMAGE_PATH]
    for path in paths:
        print(path)
        for folder in train_folders:
            #print(folder)
            #isDirectory = os.path.isdir(fpath)
            if os.path.isdir(os.path.join(path,folder)):

                folder_dir=os.path.join(path,folder)
                num_of_files=len(os.listdir(folder_dir))
                print(f"{folder}: {num_of_files}")
            
        
        
if __name__=="__main__":
    report_data()