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
from datetime import datetime

import win32gui

def callback(hwnd, extra):
    hwndMain  = win32gui.FindWindow(None, "Runelite - Blyatypus_IM")
    rect = win32gui.GetWindowRect(hwndMain)
    x = rect[0]
    y = rect[1]
    w = rect[2] - x
    h = rect[3] - y
    

# User parameters
SAVE_NAME_OD = "./Models/OSRS_Mining-0.model"
DATASET_PATH = "./Training_Data/" + SAVE_NAME_OD.split("./Models/",1)[1].split("-",1)[0] +"/"
IMAGE_SIZE              = int(re.findall(r'\d+', SAVE_NAME_OD)[-1] ) # Row and column number 
MIN_SCORE               = 0.7
TIME_BETWEEN_MINING     = 0.5 # Set 4.0 default for iron ore with mith pick

    

def most_centered_coordinates(dieCoordinates, die_class_indexes, interested_index=2):
    
    enemy_coordinates_list = dieCoordinates[die_class_indexes == interested_index].tolist() # SHOULD "== 1].tolist()" FOR ENEMY
    
    if len(enemy_coordinates_list) > 0:
        center_enemy_x_len_list = []
        center_enemy_y_len_list = []
        for enemy_coordinates in enemy_coordinates_list:
            center_enemy_x = int(enemy_coordinates[0]
                                +(enemy_coordinates[2]-enemy_coordinates[0])/2
                                )
            center_enemy_y = int(enemy_coordinates[1]
                                +(enemy_coordinates[3]-enemy_coordinates[1])/2
                                )
            center_enemy_x_len_list.append(center_enemy_x)
            center_enemy_y_len_list.append(center_enemy_y)
        
        most_centered_hypotenuse = 100000
        for index, enemy_coordinates in enumerate(enemy_coordinates_list):
            hypotenuse = sqrt(center_enemy_y_len_list[index]**2 + center_enemy_x_len_list[index]**2)
            if hypotenuse < most_centered_hypotenuse:
                most_centered_hypotenuse = hypotenuse
                most_centered_to_enemy_x = center_enemy_x_len_list[index]
                most_centered_to_enemy_y = center_enemy_y_len_list[index]
        
        x_move = most_centered_to_enemy_x + x_screen_start
        y_move = most_centered_to_enemy_y + y_screen_start
    
    return (x_move, y_move)


def cursor(x,y):
    x = int(x*2/3)
    y = int(y*2/3)
    win32api.SetCursorPos((x,y))


def left_click(x, y, time_sleep = 0):
    x = int(x*2/3)
    y = int(y*2/3)
    win32api.SetCursorPos((x,y))
    time.sleep(0.1)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)
    time.sleep(time_sleep)
    time.sleep(random.randrange(1))


def rand_spot(x_min, x_max, y_min, y_max):
    x_min = int( x_min + 0.25*(x_max-x_min) )
    x_max = int( x_max - 0.25*(x_max-x_min) )
    y_min = int( y_min + 0.25*(y_max-y_min) )
    y_max = int( y_max - 0.25*(y_max-y_min) )
    
    x_click = random.randrange(x_min, x_max)
    y_click = random.randrange(y_min, y_max)
    return x_click, y_click


def drop_inventory():
    # Takes screenshot to check size of screen
    screenshot_sizer = ImageGrab.grab()

    # Finds window size and where coordinates starts and ends in window
    x_screen_start = screenshot_sizer.size[0]-2100
    y_screen_start = 0

    # Coordinates of first inventory slot relative to screen start
    inv_start_x = 1800
    inv_start_y = 940

    # Coordinates of first inventory slot
    x_start = x_screen_start + inv_start_x
    y_start = y_screen_start + inv_start_y
    
    for y_index in range(7):
        for x_index in range(4):
            
            if x_index == 1 and y_index == 5:
                break
            
            # ------------ Comment out if not using drop plugin -------------------------
            # Left clicks for drop
            win32api.SetCursorPos((x_start+40*x_index, y_start+37*y_index))
            time.sleep(0.1)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x_start, y_start, 0, 0)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x_start, y_start, 0, 0)
            time.sleep(0.1)
            # ---------------------------------------------------------------------------
            
            # # ------------ Comment out if not using drop plugin -------------------------
            # # Right clicks for drop
            # win32api.SetCursorPos((x_start+40*x_index, y_start+37*y_index))
            # time.sleep(0.1)
            # win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, x_start, y_start, 0, 0)
            # win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, x_start, y_start, 0, 0)
            # time.sleep(0.1)
            
            # # Left clicks drop button
            # # If item is on last row, adjust height
            # if y_index == 6:
            #     win32api.SetCursorPos((x_start+40*x_index-10, y_start+37*y_index+20))
            # else:
            #     win32api.SetCursorPos((x_start+40*x_index-10, y_start+37*y_index+40))
            # time.sleep(0.1)
            # win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x_start, y_start, 0, 0)
            # win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x_start, y_start, 0, 0)
            # time.sleep(0.1)
            # # ---------------------------------------------------------------------------


def banker():
    # Takes screenshot to check size of screen
    screenshot_sizer = ImageGrab.grab()
    
    # Finds window size and where coordinates starts and ends in window
    x_screen_start = screenshot_sizer.size[0]-2100
    y_screen_start = 0
    
    # Fixes minimap
    # -------------------------------------------------------------------------
    # Coordinates of first inventory slot relative to screen start
    inv_start_x = 1800
    inv_start_y = 70
    
    x_start = x_screen_start + inv_start_x
    y_start = y_screen_start + inv_start_y
    
    left_click(x_start, y_start, time_sleep = 0.6)
    
    win32api.keybd_event(0x26, 0,0,0)
    time.sleep(0.6)
    win32api.keybd_event(0x26, 0 ,win32con.KEYEVENTF_KEYUP ,0)
    # -------------------------------------------------------------------------
    
    
    # # Runs to bank section
    # # -------------------------------------------------------------------------
    # # winsound.Beep(frequency, duration)
    # temp_screenshot = ImageGrab.grab(bbox =(screenshot_sizer.size[0]-2100, 
    #                                    0,
    #                                    screenshot_sizer.size[0], 
    #                                    screenshot_sizer.size[1]
    #                                    )
    #                             )
    
    # # temp_screenshot.save('./Images/Screenshots/image-{}.jpg'.format(ii))
    
    # screenshot_cv2 = np.array(temp_screenshot)
    # # screenshot_cv2 = cv2.cvtColor(screenshot_cv2, cv2.COLOR_BGR2RGB)
    
    # transformed_image = transforms_1(image=screenshot_cv2)
    # transformed_image = transformed_image["image"]
    
    # with torch.no_grad():
    #     prediction_1 = model_1([(transformed_image/255).to(device)])
    #     pred_1 = prediction_1[0]
    
    # dieCoordinates = pred_1['boxes'][pred_1['scores'] > MIN_SCORE]
    # die_class_indexes = pred_1['labels'][pred_1['scores'] > MIN_SCORE]
    # # BELOW SHOWS SCORES - COMMENT OUT IF NEEDED
    # die_scores = pred_1['scores'][pred_1['scores'] > MIN_SCORE]
    
    # enemy_coordinates_list = dieCoordinates[die_class_indexes == 4].tolist() 
    
    # die_class_indexes = die_class_indexes.tolist()
    # # BELOW SHOWS SCORES - COMMENT OUT IF NEEDED
    # die_scores = die_scores.tolist()
    
    # if len(enemy_coordinates_list) > 0:
    #     center_enemy_x_len_list = []
    #     center_enemy_y_len_list = []
    #     for enemy_coordinates in enemy_coordinates_list:
    #         center_enemy_x = int(enemy_coordinates[0]
    #                             +(enemy_coordinates[2]-enemy_coordinates[0])/2
    #                             )
    #         center_enemy_y = int(enemy_coordinates[1]
    #                             +(enemy_coordinates[3]-enemy_coordinates[1])/2
    #                             )
    #         center_enemy_x_len_list.append(center_enemy_x)
    #         center_enemy_y_len_list.append(center_enemy_y)
        
    #     most_centered_hypotenuse = 100000
    #     for index, enemy_coordinates in enumerate(enemy_coordinates_list):
    #         hypotenuse = sqrt(center_enemy_y_len_list[index]**2 + center_enemy_x_len_list[index]**2)
    #         if hypotenuse < most_centered_hypotenuse:
    #             most_centered_hypotenuse = hypotenuse
    #             most_centered_to_enemy_x = center_enemy_x_len_list[index]
    #             most_centered_to_enemy_y = center_enemy_y_len_list[index]
    #             index_to_use = index
        
        
    #     x_click, y_click = rand_spot(enemy_coordinates_list[index_to_use][0], 
    #                                  enemy_coordinates_list[index_to_use][2], 
    #                                  enemy_coordinates_list[index_to_use][1], 
    #                                  enemy_coordinates_list[index_to_use][3])
        
    #     x_move = x_click + x_screen_start
    #     y_move = y_click + y_screen_start
    
    # left_click(x_move, y_move, time_sleep = 4)
    # # -------------------------------------------------------------------------
    
    
    # Clicks Bank
    # -------------------------------------------------------------------------
    # winsound.Beep(frequency, duration)
    temp_screenshot = ImageGrab.grab(bbox =(screenshot_sizer.size[0]-2100, 
                                       0,
                                       screenshot_sizer.size[0], 
                                       screenshot_sizer.size[1]
                                       )
                                )
    
    # temp_screenshot.save('./Images/Screenshots/image-{}.jpg'.format(ii))
    
    screenshot_cv2 = np.array(temp_screenshot)
    # screenshot_cv2 = cv2.cvtColor(screenshot_cv2, cv2.COLOR_BGR2RGB)
    
    transformed_image = transforms_1(image=screenshot_cv2)
    transformed_image = transformed_image["image"]
    
    with torch.no_grad():
        prediction_1 = model_1([(transformed_image/255).to(device)])
        pred_1 = prediction_1[0]
    
    dieCoordinates = pred_1['boxes'][pred_1['scores'] > MIN_SCORE]
    die_class_indexes = pred_1['labels'][pred_1['scores'] > MIN_SCORE]
    # BELOW SHOWS SCORES - COMMENT OUT IF NEEDED
    die_scores = pred_1['scores'][pred_1['scores'] > MIN_SCORE]
    
    enemy_coordinates_list = dieCoordinates[die_class_indexes == 3].tolist() 
    
    die_class_indexes = die_class_indexes.tolist()
    # BELOW SHOWS SCORES - COMMENT OUT IF NEEDED
    die_scores = die_scores.tolist()
    
    if len(enemy_coordinates_list) > 0:
        center_enemy_x_len_list = []
        center_enemy_y_len_list = []
        for enemy_coordinates in enemy_coordinates_list:
            center_enemy_x = int(enemy_coordinates[0]
                                +(enemy_coordinates[2]-enemy_coordinates[0])/2
                                )
            center_enemy_y = int(enemy_coordinates[1]
                                +(enemy_coordinates[3]-enemy_coordinates[1])/2
                                )
            center_enemy_x_len_list.append(center_enemy_x)
            center_enemy_y_len_list.append(center_enemy_y)
        
        most_centered_hypotenuse = 100000
        for index, enemy_coordinates in enumerate(enemy_coordinates_list):
            hypotenuse = sqrt(center_enemy_y_len_list[index]**2 + center_enemy_x_len_list[index]**2)
            if hypotenuse < most_centered_hypotenuse:
                most_centered_hypotenuse = hypotenuse
                most_centered_to_enemy_x = center_enemy_x_len_list[index]
                most_centered_to_enemy_y = center_enemy_y_len_list[index]
                index_to_use = index
        
        
        x_click, y_click = rand_spot(enemy_coordinates_list[index_to_use][0], 
                                     enemy_coordinates_list[index_to_use][2], 
                                     enemy_coordinates_list[index_to_use][1], 
                                     enemy_coordinates_list[index_to_use][3])
        
        x_move = x_click + x_screen_start
        y_move = y_click + y_screen_start
    
    left_click(x_move, y_move, time_sleep = 5)
    # -------------------------------------------------------------------------
    
    
    # bank deposit all inventory section
    # -------------------------------------------------------------------------
    inv_start_x = 1150
    inv_start_y = 1080
    
    x_start = x_screen_start + inv_start_x
    y_start = y_screen_start + inv_start_y
    
    left_click(x_start, y_start, time_sleep = 1)
    # -------------------------------------------------------------------------
    
    
    # # Grabs Pickaxe Section
    # # -------------------------------------------------------------------------
    # temp_screenshot = ImageGrab.grab(bbox =(screenshot_sizer.size[0]-2100, 
    #                                     0,
    #                                     screenshot_sizer.size[0], 
    #                                     screenshot_sizer.size[1]
    #                                     )
    #                             )
    
    # # temp_screenshot.save('./Images/Screenshots/image-{}.jpg'.format(ii))
    
    # screenshot_cv2 = np.array(temp_screenshot)
    # # screenshot_cv2 = cv2.cvtColor(screenshot_cv2, cv2.COLOR_BGR2RGB)
    
    # transformed_image = transforms_1(image=screenshot_cv2)
    # transformed_image = transformed_image["image"]
    
    # with torch.no_grad():
    #     prediction_1 = model_1([(transformed_image/255).to(device)])
    #     pred_1 = prediction_1[0]
    
    # dieCoordinates = pred_1['boxes'][pred_1['scores'] > MIN_SCORE]
    # die_class_indexes = pred_1['labels'][pred_1['scores'] > MIN_SCORE]
    # # BELOW SHOWS SCORES - COMMENT OUT IF NEEDED
    # die_scores = pred_1['scores'][pred_1['scores'] > MIN_SCORE]
    
    # enemy_coordinates_list = dieCoordinates[die_class_indexes == 6].tolist() 
    
    # if len(enemy_coordinates_list) > 0:
    #     center_enemy_x_len_list = []
    #     center_enemy_y_len_list = []
    #     for enemy_coordinates in enemy_coordinates_list:
    #         center_enemy_x = int(enemy_coordinates[0]
    #                             +(enemy_coordinates[2]-enemy_coordinates[0])/2
    #                             )
    #         center_enemy_y = int(enemy_coordinates[1]
    #                             +(enemy_coordinates[3]-enemy_coordinates[1])/2
    #                             )
    #         center_enemy_x_len_list.append(center_enemy_x)
    #         center_enemy_y_len_list.append(center_enemy_y)
        
    #     most_centered_hypotenuse = 100000
    #     for index, enemy_coordinates in enumerate(enemy_coordinates_list):
    #         hypotenuse = sqrt(center_enemy_y_len_list[index]**2 + center_enemy_x_len_list[index]**2)
    #         if hypotenuse < most_centered_hypotenuse:
    #             most_centered_hypotenuse = hypotenuse
    #             most_centered_to_enemy_x = center_enemy_x_len_list[index]
    #             most_centered_to_enemy_y = center_enemy_y_len_list[index]
    #             index_to_use = index
        
        
    #     x_click, y_click = rand_spot(enemy_coordinates_list[index_to_use][0], 
    #                                   enemy_coordinates_list[index_to_use][2], 
    #                                   enemy_coordinates_list[index_to_use][1], 
    #                                   enemy_coordinates_list[index_to_use][3])
        
    #     x_move = x_click + x_screen_start
    #     y_move = y_click + y_screen_start
    
    # left_click(x_move, y_move, time_sleep = 1)
    
    # ALTERNATIVE
    left_click(4140, 720+55*4, time_sleep = 1)
    # -------------------------------------------------------------------------
    
    
    # Runs back section
    # -------------------------------------------------------------------------
    inv_start_x = 4985 # Iron: 1965; Coal: 4970
    inv_start_y = 150 # Iron: 150; Coal: 188
    
    x_start = x_screen_start + inv_start_x - x_screen_start
    y_start = y_screen_start + inv_start_y - y_screen_start
    
    
    left_click(x_start, y_start, time_sleep = 5)
    # -------------------------------------------------------------------------


def banker_varrock():
    # Takes screenshot to check size of screen
    screenshot_sizer = ImageGrab.grab()
    
    # Finds window size and where coordinates starts and ends in window
    x_screen_start = screenshot_sizer.size[0]-2100
    y_screen_start = 0
    
    # Fixes minimap
    # -------------------------------------------------------------------------
    # Coordinates of first inventory slot relative to screen start
    inv_start_x = 1800
    inv_start_y = 70
    
    x_start = x_screen_start + inv_start_x
    y_start = y_screen_start + inv_start_y
    
    left_click(x_start, y_start, time_sleep = 0.1)
    
    win32api.keybd_event(0x26, 0,0,0)
    time.sleep(0.6)
    win32api.keybd_event(0x26, 0 ,win32con.KEYEVENTF_KEYUP ,0)
    # -------------------------------------------------------------------------
    
    
    # Runs to bank section
    # -------------------------------------------------------------------------
    # Coordinates of first inventory slot relative to screen start
    inv_start_x = 1820+100+40
    inv_start_y = 60+5
    
    x_start = x_screen_start + inv_start_x
    y_start = y_screen_start + inv_start_y
    
    left_click(x_start, y_start, 13)
    
    # Coordinates of first inventory slot relative to screen start
    inv_start_x = 1820+100
    inv_start_y = 60+5
    
    x_start = x_screen_start + inv_start_x
    y_start = y_screen_start + inv_start_y
    
    left_click(x_start, y_start, 13)
    
    left_click(x_start, y_start, 13)
    
    # Coordinates of first inventory slot relative to screen start
    inv_start_x = 1820+10
    inv_start_y = 60+5+50
    
    x_start = x_screen_start + inv_start_x
    y_start = y_screen_start + inv_start_y
    
    left_click(x_start, y_start, 13)
    
    # Coordinates of first inventory slot relative to screen start
    inv_start_x = 1820
    inv_start_y = 60+5+120
    
    x_start = x_screen_start + inv_start_x
    y_start = y_screen_start + inv_start_y
    
    left_click(x_start, y_start, 14)
    
    # Coordinates of first inventory slot relative to screen start
    inv_start_x = 1820+50
    inv_start_y = 60+5+150
    
    x_start = x_screen_start + inv_start_x
    y_start = y_screen_start + inv_start_y
    
    left_click(x_start, y_start, 13)
    # -------------------------------------------------------------------------
    
    
    # Clicks banks
    # -------------------------------------------------------------------------
    # Coordinates of first inventory slot relative to screen start
    inv_start_x = 1050
    inv_start_y = 800
    
    x_start = x_screen_start + inv_start_x
    y_start = y_screen_start + inv_start_y
    
    left_click(x_start, y_start, 2)
    # -------------------------------------------------------------------------
    
    
    # bank deposit all inventory section
    # -------------------------------------------------------------------------
    inv_start_x = 1150
    inv_start_y = 1080
    
    x_start = x_screen_start + inv_start_x
    y_start = y_screen_start + inv_start_y
    
    
    left_click(x_start, y_start, 1)
    
    
    # Runs back section
    # -------------------------------------------------------------------------
    # Coordinates of first inventory slot relative to screen start
    inv_start_x = 1820+180
    inv_start_y = 60+60
    
    x_start = x_screen_start + inv_start_x
    y_start = y_screen_start + inv_start_y
    
    left_click(x_start, y_start, 13)
    
    # Coordinates of first inventory slot relative to screen start
    inv_start_x = 1820+180
    inv_start_y = 200
    
    x_start = x_screen_start + inv_start_x
    y_start = y_screen_start + inv_start_y
    
    left_click(x_start, y_start, 13)
    
    # Coordinates of first inventory slot relative to screen start
    inv_start_x = 1820+140
    inv_start_y = 260
    
    x_start = x_screen_start + inv_start_x
    y_start = y_screen_start + inv_start_y
    
    left_click(x_start, y_start, 13)
    
    # Coordinates of first inventory slot relative to screen start
    inv_start_x = 1820+100
    inv_start_y = 260
    
    x_start = x_screen_start + inv_start_x
    y_start = y_screen_start + inv_start_y
    
    left_click(x_start, y_start, 13)
    
    left_click(x_start, y_start, 13)
    
    # Coordinates of first inventory slot relative to screen start
    inv_start_x = 1820+50
    inv_start_y = 210
    
    x_start = x_screen_start + inv_start_x
    y_start = y_screen_start + inv_start_y
    
    left_click(x_start, y_start, 13)
    # -------------------------------------------------------------------------


def trader_drawf_mine():
    # Takes screenshot to check size of screen
    screenshot_sizer = ImageGrab.grab()
    
    # Finds window size and where coordinates starts and ends in window
    x_screen_start = screenshot_sizer.size[0]-2100
    y_screen_start = 0
    
    # Fixes minimap
    # -------------------------------------------------------------------------
    # Coordinates of first inventory slot relative to screen start
    inv_start_x = 1800
    inv_start_y = 70
    
    x_start = x_screen_start + inv_start_x
    y_start = y_screen_start + inv_start_y
    
    
    left_click(x_start, y_start, time_sleep = 0.1)
    
    win32api.keybd_event(0x26, 0,0,0)
    time.sleep(0.6)
    win32api.keybd_event(0x26, 0 ,win32con.KEYEVENTF_KEYUP ,0)
    # -------------------------------------------------------------------------
    
    
    # Runs to trader dwarf section
    # -------------------------------------------------------------------------
    # Coordinates of first inventory slot relative to screen start
    inv_start_x = 1800+140
    inv_start_y = 60
    
    x_start = x_screen_start + inv_start_x
    y_start = y_screen_start + inv_start_y
    
    left_click(x_start, y_start, 20)
    # -------------------------------------------------------------------------
    
    
    # Clicks banks
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    
    
    # bank deposit all inventory section
    # -------------------------------------------------------------------------
    inv_start_x = 1150
    inv_start_y = 1080
    
    x_start = x_screen_start + inv_start_x
    y_start = y_screen_start + inv_start_y
    
    
    left_click(x_start, y_start, 1)
    
    
    # Runs back section
    # -------------------------------------------------------------------------
    # Coordinates of first inventory slot relative to screen start
    inv_start_x = 1820+180
    inv_start_y = 60+60
    
    x_start = x_screen_start + inv_start_x
    y_start = y_screen_start + inv_start_y
    
    left_click(x_start, y_start, 13)
    
    # Coordinates of first inventory slot relative to screen start
    inv_start_x = 1820+180
    inv_start_y = 200
    
    x_start = x_screen_start + inv_start_x
    y_start = y_screen_start + inv_start_y
    
    left_click(x_start, y_start, 13)
    
    # Coordinates of first inventory slot relative to screen start
    inv_start_x = 1820+140
    inv_start_y = 260
    
    x_start = x_screen_start + inv_start_x
    y_start = y_screen_start + inv_start_y
    
    left_click(x_start, y_start, 13)
    
    # Coordinates of first inventory slot relative to screen start
    inv_start_x = 1820+100
    inv_start_y = 260
    
    x_start = x_screen_start + inv_start_x
    y_start = y_screen_start + inv_start_y
    
    left_click(x_start, y_start, 13)
    
    left_click(x_start, y_start, 13)
    
    # Coordinates of first inventory slot relative to screen start
    inv_start_x = 1820+50
    inv_start_y = 210
    
    x_start = x_screen_start + inv_start_x
    y_start = y_screen_start + inv_start_y
    
    left_click(x_start, y_start, 13)
    # -------------------------------------------------------------------------


def mining(x_screen_start, y_screen_start, ii, stop_index, 
           num_ore_collected, force_mine, force_mine_count):
    
    prev_num_ore_collected = num_ore_collected
    
    
    # Searches for number of ores collected
    # -------------------------------------------------------------------------
    # winsound.Beep(frequency, duration)
    temp_screenshot = ImageGrab.grab(bbox =(screenshot_sizer.size[0]-450, 
                                       850,
                                       screenshot_sizer.size[0], 
                                       screenshot_sizer.size[1]
                                       )
                                )
    
    # temp_screenshot.save('./Images/Screenshots/image-{}.jpg'.format(ii))
    
    screenshot_cv2 = np.array(temp_screenshot)
    # screenshot_cv2 = cv2.cvtColor(screenshot_cv2, cv2.COLOR_BGR2RGB)
    
    transformed_image = transforms_1(image=screenshot_cv2)
    transformed_image = transformed_image["image"]
    
    with torch.no_grad():
        prediction_1 = model_1([(transformed_image/255).to(device)])
        pred_1 = prediction_1[0]
    
    dieCoordinates = pred_1['boxes'][pred_1['scores'] > MIN_SCORE]
    die_class_indexes = pred_1['labels'][pred_1['scores'] > MIN_SCORE]
    
    ore_collected_list = dieCoordinates[die_class_indexes == 8].tolist()
    
    num_ore_collected = len(ore_collected_list)
    # -------------------------------------------------------------------------
    
    
    # Looks at ores
    # -------------------------------------------------------------------------
    # winsound.Beep(frequency, duration)
    temp_screenshot = ImageGrab.grab(bbox =(screenshot_sizer.size[0]-1250, 
                                       380,
                                       screenshot_sizer.size[0]-350-400, 
                                       screenshot_sizer.size[1]-100
                                       )
                                )
    
    # temp_screenshot.save('./Images/Screenshots/image-{}.jpg'.format(ii+1000))
    
    screenshot_cv2 = np.array(temp_screenshot)
    # screenshot_cv2 = cv2.cvtColor(screenshot_cv2, cv2.COLOR_BGR2RGB)
    
    transformed_image = transforms_1(image=screenshot_cv2)
    transformed_image = transformed_image["image"]
    
    with torch.no_grad():
        prediction_1 = model_1([(transformed_image/255).to(device)])
        pred_1 = prediction_1[0]
    
    dieCoordinates = pred_1['boxes'][pred_1['scores'] > MIN_SCORE]
    die_class_indexes = pred_1['labels'][pred_1['scores'] > MIN_SCORE]
    
    enemy_coordinates_list = dieCoordinates[die_class_indexes == 9].tolist()
    # -------------------------------------------------------------------------
    
    # Searches if still mining
    if ((num_ore_collected == prev_num_ore_collected 
         or len(enemy_coordinates_list) == 0
         )
        and not force_mine
        ):
        time.sleep(1)
        
        force_mine_count += 1
        
        if force_mine_count >= 3:
            force_mine = True
            force_mine_count = 0
        
        return stop_index, num_ore_collected, force_mine, force_mine_count
    
    if len(enemy_coordinates_list) > 0:
        center_enemy_x_len_list = []
        center_enemy_y_len_list = []
        for enemy_coordinates in enemy_coordinates_list:
            center_enemy_x = int(enemy_coordinates[0]
                                +(enemy_coordinates[2]-enemy_coordinates[0])/2
                                )
            center_enemy_y = int(enemy_coordinates[1]
                                +(enemy_coordinates[3]-enemy_coordinates[1])/2
                                )
            center_enemy_x_len_list.append(center_enemy_x)
            center_enemy_y_len_list.append(center_enemy_y)
        
        most_centered_hypotenuse = 100000
        index_to_use = 999
        for index, enemy_coordinates in enumerate(enemy_coordinates_list):
            hypotenuse = sqrt(center_enemy_y_len_list[index]**2 + center_enemy_x_len_list[index]**2)
            if hypotenuse < most_centered_hypotenuse:
                most_centered_hypotenuse = hypotenuse
                most_centered_to_enemy_x = center_enemy_x_len_list[index]
                most_centered_to_enemy_y = center_enemy_y_len_list[index]
                index_to_use = index
        
        
        x_click, y_click = rand_spot(enemy_coordinates_list[index_to_use][0], 
                                     enemy_coordinates_list[index_to_use][2], 
                                     enemy_coordinates_list[index_to_use][1], 
                                     enemy_coordinates_list[index_to_use][3])
        
        # x_move = most_centered_to_enemy_x + x_screen_start
        # y_move = most_centered_to_enemy_y + y_screen_start
        x_move = x_click + x_screen_start
        y_move = y_click + y_screen_start
        
        # x_move = center_enemy_x_len_list[0] + x_screen_start
        # y_move = center_enemy_y_len_list[0] + y_screen_start
        
        left_click(x_move, y_move, time_sleep = TIME_BETWEEN_MINING)
        
        force_mine = False
        force_mine_count = 0
        
        return stop_index, num_ore_collected, force_mine, force_mine_count
    else:
        stop_index += 1
        force_mine = False
        
        if stop_index == 20:
            print("Stopping!")

            # datetime object containing current date and time
            now = datetime.now()
             
            print("now =", now)
            
            # dd/mm/YY H:M:S
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            print("date and time =", dt_string)	
            
            sys.exit()
        
        return stop_index, num_ore_collected, force_mine, force_mine_count



# Main()
dataset_path = DATASET_PATH

# Windows beep settings
frequency = 700  # Set Frequency To 2500 Hertz
duration = 80  # Set Duration To 1000 ms == 1 second



#load classes
coco = COCO(os.path.join(dataset_path, "train", "_annotations.coco.json"))
categories = coco.cats
n_classes_1 = len(categories.keys())
categories

classes_1 = [i[1]['name'] for i in categories.items()]
classes_1



# lets load the faster rcnn model
model_1 = models.detection.fasterrcnn_resnet50_fpn(pretrained=True, 
                                                   min_size=1400,
                                                   max_size=2500)
in_features = model_1.roi_heads.box_predictor.cls_score.in_features # we need to change the head
model_1.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes_1)

# Loads last saved checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'

if os.path.isfile(SAVE_NAME_OD):
    checkpoint = torch.load(SAVE_NAME_OD, map_location=map_location)
    model_1.load_state_dict(checkpoint)

model_1 = model_1.to(device)

model_1.eval()
torch.cuda.empty_cache()

transforms_1 = A.Compose([
    # A.Resize(IMAGE_SIZE, IMAGE_SIZE), # our input size can be 600px
    # A.Rotate(limit=[90,90], always_apply=True),
    ToTensorV2()
])


# Takes screenshot to check size of screen
screenshot_sizer = ImageGrab.grab()

# Finds window size and where coordinates starts and ends in window
x_screen_start = screenshot_sizer.size[0]-1250
y_screen_start = 380


while True:
    stop_index = 0
    num_ore_collected = 99
    force_mine = True
    force_mine_count = 0
    ii = 0
    while True:
        (stop_index, num_ore_collected, 
         force_mine, force_mine_count) = mining(x_screen_start, 
                                                y_screen_start, 
                                                ii, 
                                                stop_index,
                                                num_ore_collected,
                                                force_mine,
                                                force_mine_count
                                                )
        if num_ore_collected >= 24:
            break
        ii += 1
    
    banker()
    
    # banker_varrock()
    
    # drop_inventory()

# win32api.SetCursorPos((x_screen_start+1025, y_screen_start+625))
# win32api.SetCursorPos((x_screen_start+950, y_screen_start+700))




