import os
import torch
from torchvision import models
import re
import cv2
import albumentations as A  # our data augmentation library
# remove arnings (optional)
import warnings
warnings.filterwarnings("ignore")
import time
from pycocotools.coco import COCO
# Now, we will define our transforms
from albumentations.pytorch import ToTensorV2
from torchvision.utils import draw_bounding_boxes
import shutil
from PIL import ImageGrab
import winsound
import random
import win32api, win32con
import numpy as np
import time
from math import sqrt
from torchvision.utils import save_image
import sys
import win32gui
    

# User parameters
SAVE_NAME_OD = "./Models/OSRS_Mining-0.model"
DATASET_PATH = "./Training_Data/" + SAVE_NAME_OD.split("./Models/",1)[1].split("-",1)[0] +"/"
IMAGE_SIZE              = int(re.findall(r'\d+', SAVE_NAME_OD)[-1] ) # Row and column number 
MIN_SCORE               = 0.7
TIME_BETWEEN_MINING     = 4 # Set 2.0 default for one pick iron


def cursor(x,y):
    win32api.SetCursorPos((x,y))


def left_click(x, y, time_sleep = 0):
    win32api.SetCursorPos((x,y))
    time.sleep(0.1)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)
    time.sleep(time_sleep)


def fix_minimap():
    
    left_click(int(4820*2/3), int(70*2/3), time_sleep = 0.6)
    
    win32api.keybd_event(0x26, 0,0,0)
    time.sleep(0.6)
    win32api.keybd_event(0x26, 0 ,win32con.KEYEVENTF_KEYUP ,0)


def baker_clicker():
    left_click(int(3900*2/3), int(650*2/3), time_sleep = 3.5)


def banker():
    # Walk to the bank
    # -------------------------------------------------------------------------
    left_click(int(4900*2/3), int(250*2/3), time_sleep = 12)
    left_click(int(4900*2/3), int(225*2/3), time_sleep = 28)
    # -------------------------------------------------------------------------
    
    # Clicks Bank
    # -------------------------------------------------------------------------
    left_click(int(4180*2/3), int(670*2/3), time_sleep = 1.5)
    # -------------------------------------------------------------------------
    
    # Deposits All
    # -------------------------------------------------------------------------
    left_click(int(4170*2/3), int(1080*2/3), time_sleep = 1)
    # -------------------------------------------------------------------------
    
    # Walks back
    # -------------------------------------------------------------------------
    left_click(int(5010*2/3), int(100*2/3), time_sleep = 28)
    left_click(int(4945*2/3), int(85*2/3), time_sleep = 12)
    # -------------------------------------------------------------------------



# Main()
# ==============================================================================


for i in range(1000):
    for ii in range(28):
        baker_clicker()
    
    fix_minimap()
    
    banker()




