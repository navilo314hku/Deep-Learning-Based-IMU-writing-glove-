from const import *
import sys
from datetime import datetime
import numpy as np 
import cv2
import os 
from time import sleep
def storeTxtToJpg(TXT_PATH,IMAGE_PATH,label):
    def arrayFromFile(file_name):
        arr=[]
        with open(file_name,'r') as f:
            for line in f.readlines(): #line is a string
                line_arr=[float(x) for x in line.split(",")]
                arr.append(line_arr)
        arr=np.array(arr)
        return arr
    def isEmptyTxt(file_name):
        return os.stat(f"{file_name}").st_size == 0
    def currentTimeInfo():
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    #convert all txt files from {TXT_PATH} into jpg images and store them in {IMAGE_PATH}
    #1. list all files name with txt extension
    destination_folder= os.path.join(IMAGE_PATH,f"{label}")
    for file in os.listdir(TXT_PATH):
        if file.endswith(".txt"):
            file_name=os.path.join(TXT_PATH,file)
            if not isEmptyTxt(file_name): 
                print(file_name)
                img_array=arrayFromFile(file_name)
                sleep(0.8)
                #cv2.imwrite(img_)
                #path=os.path.join("images","your_file.jpg")
                output_path=os.path.join(destination_folder,currentTimeInfo()+".jpg")
                cv2.imwrite(output_path,img_array)

if __name__=='__main__':
    if len(sys.argv)!=2:
        print("NO LABEL ARGUMENT!!!")
        print("QUITTED")
        quit()
    else:
        #print("the argument is ")
        #print(sys.argv[1])
        storeTxtToJpg(TXT_PATH,IMAGE_PATH,sys.argv[1])
    