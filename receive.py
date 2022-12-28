# This program include 
# 1. receiving data from USB by pyserial
# 2. Normalize the data between 0 and 1 
# 3. Store the data in image/ directory in .jpg format
REMOVE=True
PRINT_DATA=True


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
def mainLoop(serialInst):
    if (REMOVE):
        removeTxt()

    count=1
    txtIndx=0
    writing=False
    while True: 
        #print("running")
        if serialInst.in_waiting:
            packet=serialInst.readline()
            txt=(packet.decode('ISO-8859-1'))
            if PRINT_DATA:
                print(txt)
            file_name=os.path.join(TXT_PATH,f"{txtIndx}.txt")
            with open(file_name, 'a') as f:
                if isStart(txt) and not (writing):
                    writing=True
                    startTime=time.time()
                    print("Start detected")
                if writing==True:
                    #print("writing True")
                    if  isStart(txt)==0 and isSave(txt)==0:#IMU data
                        if count<=SAMPLE_SIZE:
                            f.write(txt)
                        if count==SAMPLE_SIZE:
                            txtIndx+=1
                            count=0
                            writing=False
                            print(time.time()-startTime)
                        count+=1
serialInst=portSetup()            
mainLoop(serialInst)