from const import *
from datetime import datetime
from matplotlib import image
import numpy as np 
import cv2
import os 
from time import sleep
from models.customFunctions import *
import json
import argparse
def getReceivePyParserArgument():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--type",help="model datatype of receive: u for unfixed, f for fixed")
    #parser.add_argument("--times",help="number of times you want to print the echo arg",type=int)
    args = parser.parse_args()
    if not args.type:
        raise Exception("missing datatype: u/f")
        quit()
    
    return args
class ConfJsonDictAccesser():
    class DataLengthType():
        fix='f'
        unfix='u'
    def get_dict(self):
        with open(CONF_JSON_PATH, 'r') as openfile:
        # Reading from json file
            json_object = json.load(openfile)
            return json_object
        
    def get_model_data_type(self):
        dict=self.get_dict()
        return dict["modelDataType"]
    
    def writeDataLengthType(self,mode):
        if mode!="f" and mode!="u":
            raise Exception("no such mode in conf.json")
            quit()
        
        JSON_dict={
            "modelDataType":mode
        }
        JSON_obj=json.dumps(JSON_dict,indent=4)
        with open(CONF_JSON_PATH,"w") as jsonFile:
            jsonFile.write(JSON_obj)
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
    JsonAc=ConfJsonDictAccesser()
    paths=''
    if (JsonAc.get_model_data_type()=='f'):
        
        paths=[FIXED_LENGTH_TRAIN_PATH,FIXED_LENGTH_TEST_PATH]
    elif (JsonAc.get_model_data_type()=='u'):
        paths=[VARIED_LENGTH_TRAIN_PATH,VARIED_LENGTH_TEST_PATH]
    else:
        raise Exception("unknown data_type")
        quit()
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
    print("preparing dataset...........................")
    #get the conf json 
    JsonAc=ConfJsonDictAccesser()
    ModelDataType=JsonAc.get_model_data_type()
    if ModelDataType=='f':
        train_dataset=torchvision.datasets.ImageFolder(FIXED_LENGTH_TRAIN_PATH,transform=basicTransform,loader=custom_pil_loader)
        test_dataset= torchvision.datasets.ImageFolder(FIXED_LENGTH_TEST_PATH,transform=basicTransform,loader=custom_pil_loader)
    elif ModelDataType=='u':
        train_dataset=torchvision.datasets.ImageFolder(VARIED_LENGTH_TRAIN_PATH,transform=basicTransform,loader=custom_pil_loader)
        test_dataset= torchvision.datasets.ImageFolder(VARIED_LENGTH_TEST_PATH,transform=basicTransform,loader=custom_pil_loader)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=True, drop_last=True)
    print("dataset information: ")
    print(f"batch_size={batch_size}")
    print(f"train_dataset shape: {train_dataset[0][0].shape}")
    
    return train_dataset,test_dataset,train_loader,test_loader

def showImg0_255(img_dir):
    #print(FIXED_LENGTH_TEST_PATH)
    image_path_dict={0:"20221228_191608.jpg",
                      1:"20230102_014555.jpg",
                      2:"20221228_191744.jpg",
                      3:"20221228_191837.jpg",
                      4:"20221228_191928.jpg",
                      5:"20221228_192020.jpg",
                      6:"20221228_192105.jpg",
                      7:"20221228_192209.jpg",
                      8:"20221228_192309.jpg",
                      9:"20221228_192401.jpg"
                    
    }
    storePath=os.path.join("images","sampleImages")
    for index in image_path_dict:
        path=os.path.join(BACKUP_PATH,"test",str(index),image_path_dict[index])
        arr=image.imread(path)
        new_arr = ((arr - arr.min()) * (1/(arr.max() - arr.min()) * 255)).astype('uint8')
        file_name=f"{index}.jpg"
        file_name=os.path.join(storePath,file_name)
        cv2.imwrite(file_name,new_arr)
        print(path)
    #path=os.path.join(BACKUP_PATH,"test","1","20230102_014555.jpg")
    #print(path) 
    # arr=image.imread(path)
    # print(type(arr))
    # new_arr = ((arr - arr.min()) * (1/(arr.max() - arr.min()) * 255)).astype('uint8')
    # cv2.imwrite("1.jpg",new_arr)
    
if __name__=="__main__":
    report_data()
    #showImg0_255("")
    #JsonAc=ConfJsonDictAccesser()
    #print(JsonAc.get_model_data_type())
    #train_ds,test_dataset,_,_=getDatasetDataloader()
    #print(test_dataset.classes)