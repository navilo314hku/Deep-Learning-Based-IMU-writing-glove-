# This program include 
# 1. receiving data from USB by pyserial
# 2. Normalize the data between 0 and 1 
# 3. Store the data in image/ directory in .jpg format
PRINT_DATA=True
from models.model_classes.convNet import *
from txtToJpg import *
from test import * 
from const import * 
from utils import removeTxt
from time import sleep
import serial.tools.list_ports
import time
import os
def portSetup():
    ports= serial.tools.list_ports.comports(0)
    serialInst=serial.Serial()
    portList=[] 
    #serialPort="/dev/cu.usbserial-0001 - CP2102 USB to UART Bridge Controller - CP2102 USB to UART Bridge Controller"
    for onePort in ports:
        portList.append(str(onePort))
        print(str(onePort))

    serialInst.baudrate=BAUD_RATE
    serialInst.port=SERIAL_PORT
    serialInst.open()
    return serialInst
def isStart(txt):
    if len(txt)==7:
        return 1
    return 0
def isSave(txt):
    if len(txt)==6:
        return 1
    return 0
#main receive loop

def FixedLengthData_RealTimePredict(serialInst):
    #local const
    realTimePredictionDir=realTimePrediction.ROOT_PATH
    txt_file_name=os.path.join(realTimePredictionDir,"0.txt")
    image_file_name=os.path.join(realTimePredictionDir,"0.jpg")
    #local variable
    count=1
    txtIndx=0
    writing=False
    
    #os.remove(txt_file_name)
    #os.remove(image_file_name)
    while True: 
        #print("running")
        if serialInst.in_waiting:
            packet=serialInst.readline()
            txt=(packet.decode('ISO-8859-1'))
            if PRINT_DATA:
                print(txt)
           
            with open(txt_file_name, 'a') as f:
                if isStart(txt) and not (writing):
                    writing=True
                    startTime=time.time()
                    print("Start detected")
                if writing==True:
                    #print("writing True")
                    if  isStart(txt)==0 and isSave(txt)==0:#IMU data
                        if count<=SAMPLE_SIZE+1:
                            f.write(txt)
                        if count==SAMPLE_SIZE+1:
                            txtIndx+=1
                            count=0
                            writing=False
                            print(time.time()-startTime)
                            print(TRASH_STRING)
                            #convert txt to image
                            #Problem with storeTxtToJpg
                            storeTxtToJpg(realTimePredictionDir,realTimePredictionDir,label="",mode="prediction")
                            model_path=os.path.join(Models.TRAINED_MODELS_PATH,"OptimConvNet2_20230113_151355")
                            model=OptimConvNet2(output_size=10)
                            model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
                            img_path=os.path.join(realTimePrediction.ROOT_PATH,"0.jpg")
                            print(predictSingleImage(img_path,model=model))
                            #prediction on the image
                            #remove 0.txt
                            os.remove(txt_file_name)
                            
                            sleep(2)
                        count+=1
if __name__=="__main__":     
    serialInst=portSetup()            
    FixedLengthData_RealTimePredict(serialInst)