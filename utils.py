from const import *
from datetime import datetime
import numpy as np 
import cv2
import os 
from time import sleep

def removeTxt():
    import os
    txt_dir = os.path.join("/Users/ivanlo/Desktop/FYP/code/main/",TXT_PATH)
    test = os.listdir(txt_dir)
    for item in test:
        if item.endswith(".txt"):
            os.remove(os.path.join(txt_dir, item))

