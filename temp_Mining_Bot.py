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
import shutil
from PIL import ImageGrab
import winsound
import random
import win32api, win32con
import numpy as np
import time


def click(x,y):
    win32api.SetCursorPos((x,y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)


def drop_inventory():
    # Takes screenshot to check size of screen
    screenshot_sizer = ImageGrab.grab()

    # Finds window size and where coordinates starts and ends in window
    x_screen_start = screenshot_sizer.size[0]-2100
    y_screen_start = 0
    x_screen_stop = screenshot_sizer.size[0]
    y_screen_stop = screenshot_sizer.size[1]

    # Coordinates of first inventory slot relative to screen start
    inv_start_x = 1800
    inv_start_y = 940

    # Coordinates of first inventory slot
    x_start = int((x_screen_start + inv_start_x)*2/3)
    y_start = int((y_screen_start + inv_start_y)*2/3)
    
    for y_index in range(7):
        for x_index in range(4):
            if x_index == 3 and y_index == 6:
                break
            win32api.SetCursorPos((x_start+40*x_index, y_start+37*y_index))
            time.sleep(0.1)
            win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, x_start, y_start, 0, 0)
            win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, x_start, y_start, 0, 0)
            time.sleep(0.1)
            if y_index == 6:
                win32api.SetCursorPos((x_start+40*x_index-10, y_start+37*y_index+20))
            else:
                win32api.SetCursorPos((x_start+40*x_index-10, y_start+37*y_index+40))
            time.sleep(0.1)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x_start, y_start, 0, 0)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x_start, y_start, 0, 0)
            time.sleep(0.1)
    


def mining():
    # Takes screenshot to check size of screen
    screenshot_sizer = ImageGrab.grab()
    
    # Finds window size and where coordinates starts and ends in window
    x_screen_start = screenshot_sizer.size[0]-2100
    y_screen_start = 0
    
    x_temp_1 = int((x_screen_start+1025)*2/3)
    y_temp_1 = int((y_screen_start+625)*2/3)
    x_temp_2 = int((x_screen_start+950)*2/3)
    y_temp_2 = int((y_screen_start+700)*2/3)
    
    time_set = 3
    
    for i in range(9):
        win32api.SetCursorPos((x_temp_1+10, y_temp_1))
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x_temp_1, y_temp_1, 0, 0)
        time.sleep(0.1)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x_temp_1, y_temp_1, 0, 0)
        time.sleep(0.01)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x_temp_1, y_temp_1, 0, 0)
        time.sleep(0.1)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x_temp_1, y_temp_1, 0, 0)
        time.sleep(time_set)
        time.sleep(random.randrange(2))
        
        win32api.SetCursorPos((x_temp_2, y_temp_2))
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x_temp_2, y_temp_2, 0, 0)
        time.sleep(0.1)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x_temp_2, y_temp_2, 0, 0)
        time.sleep(0.01)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x_temp_2, y_temp_2, 0, 0)
        time.sleep(0.1)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x_temp_2, y_temp_2, 0, 0)
        time.sleep(time_set)
        time.sleep(random.randrange(2))
        
        win32api.SetCursorPos((x_temp_1+10, y_temp_1+120))
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x_temp_2, y_temp_2, 0, 0)
        time.sleep(0.1)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x_temp_2, y_temp_2, 0, 0)
        time.sleep(0.01)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x_temp_2, y_temp_2, 0, 0)
        time.sleep(0.1)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x_temp_2, y_temp_2, 0, 0)
        time.sleep(time_set)
        time.sleep(random.randrange(2))

for i in range(100):
    mining()
    
    drop_inventory()

# win32api.SetCursorPos((int((x_screen_start+1025)*2/3), int((y_screen_start+625)*2/3)))
# win32api.SetCursorPos((int((x_screen_start+950)*2/3), int((y_screen_start+700)*2/3)))




