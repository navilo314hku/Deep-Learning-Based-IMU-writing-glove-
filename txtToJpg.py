from const import *
import sys
from datetime import datetime
import numpy as np 
import cv2
import os 
from time import sleep
from threading import *
from utils import *

def removeTxt(txtBuffer):
    import os
    txtFileList = os.listdir(txtBuffer)
    for item in txtFileList:
        if item.endswith(".txt"):
            os.remove(os.path.join(txtBuffer, item))


def storeTxtToJpg(TXT_PATH,IMAGE_PATH,label,mode="dataCollection"):
    """
    TXT_PATH: folder name of the txt files
    IMAGE_PATH: destinated image folder to save, DO NOT include filename like .jpg
    mode: 'dataCollection' or 'prediction'
    
    """
    def arrayFromFile(file_name):
        arr=[]
        status=-1
        try:
            with open(file_name,'r') as f:
                for line in f.readlines(): #line is a string
                    line_arr=[float(x) for x in line.split(",")]
                    arr.append(line_arr)
            arr=np.array(arr)
            status=True
        except:
            status=False
        return arr, status
    def isEmptyTxt(file_name):
        return os.stat(f"{file_name}").st_size == 0
    def currentTimeInfo():
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    #convert all txt files from {TXT_PATH} into jpg images and store them in {IMAGE_PATH}
    #1. list all files name with txt extension
    if mode=="dataCollection":
        destination_folder= os.path.join(IMAGE_PATH,f"{label}")
        for file in os.listdir(TXT_PATH):
            if file.endswith(".txt"):
                file_name=os.path.join(TXT_PATH,file)
                if not isEmptyTxt(file_name): 
                    print(file_name)
                    img_array,result=arrayFromFile(file_name)
                    if (result==False or result==-1):
                        continue
                    if (result==-1):
                        raise Exception("error occur in arrayFromFile()")
                    sleep(1)

                    #cv2.imwrite(img_)
                    #path=os.path.join("images","your_file.jpg")
                    output_path=os.path.join(destination_folder,currentTimeInfo()+".jpg")
                    print(f"output_path={output_path}")
                    cv2.imwrite(output_path,img_array)
                    #print("complete image writing")
    elif mode=="prediction":
        destination_folder= os.path.join(IMAGE_PATH,f"{label}")
        for file in os.listdir(TXT_PATH):
            if file.endswith(".txt"):
                file_name=os.path.join(TXT_PATH,file)
                if not isEmptyTxt(file_name): 
                    print(file_name)
                    img_array=arrayFromFile(file_name)
                    sleep(1)

                    #cv2.imwrite(img_)
                    #path=os.path.join("images","your_file.jpg")
                    output_path=os.path.join(destination_folder,"0.jpg")
                    print(f"output_path={output_path}")
                    cv2.imwrite(output_path,img_array)
                    print("complete image writing") 
    else: 
        raise Exception("INVALID MODE IN storeTxtToJpg()")
        quit()
def moveToBuffer(FromPath,ToPath):
    print("moving files from txtStorage to txtBuffer")
    source = FromPath
    destination = ToPath

    # gather all files
    allfiles = os.listdir(source)

    # iterate on all files to move them to destination folder
    for f in allfiles:
        src_path = os.path.join(source, f)
        dst_path = os.path.join(destination, f)
        os.rename(src_path, dst_path)

if __name__=='__main__':
 
    if len(sys.argv)!=3:#test 0
        raise Exception("""MISSING ARGUMENT(S)!!! 
        python3 txtToJpg.py [train/test] [label]
        """)
        print("QUITTED")
        quit()
    if not (sys.argv[1]=='test' or sys.argv[1]=='train'):
        print("NO SUCH MODE, PLEASE ENTER test or train as mode")
        print("QUIT")
        quit()
        
    jsonAccesser=ConfJsonDictAccesser()
    conf_json_dict=jsonAccesser.get_dict()
    modelDataType=conf_json_dict["modelDataType"]

    if sys.argv[1]=='test':
        if modelDataType==jsonAccesser.DataLengthType.fix:
            path=FIXED_LENGTH_TEST_PATH
        elif modelDataType==jsonAccesser.DataLengthType.unfix:
            path=VARIED_LENGTH_TEST_PATH
    elif sys.argv[1]=='train':
        if modelDataType==jsonAccesser.DataLengthType.fix:
            path=FIXED_LENGTH_TRAIN_PATH
        elif modelDataType==jsonAccesser.DataLengthType.unfix:
            path=VARIED_LENGTH_TRAIN_PATH

    moveToBuffer(TXT_PATH,TXT_BUFFER)
    storeTxtToJpg(TXT_BUFFER,path,sys.argv[2])
    removeTxt(TXT_BUFFER)
    
