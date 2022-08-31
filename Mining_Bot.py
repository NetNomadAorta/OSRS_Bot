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
TIME_BETWEEN_MINING     = 4 # Set 2.0 default for one pick iron


# def predicter(
#               ):
    
#     # screenshot_x1 = screenshot_sizer.size[0]-2100, 
#     # screenshot_y1 = 0, 
#     # screenshot_x2 = screenshot_sizer.size[0], 
#     # screenshot_y2 = screenshot_sizer.size[1]
    
#     screenshot_sizer = ImageGrab.grab()
    
#     # winsound.Beep(frequency, duration)
#     temp_screenshot = ImageGrab.grab(bbox =(screenshot_x1, 
#                                             screenshot_y1,
#                                             screenshot_x2, 
#                                             screenshot_y2
#                                             )
#                                      )
    
#     # temp_screenshot.save('./Images/Screenshots/image-{}.jpg'.format(ii))
    
#     screenshot_cv2 = np.array(temp_screenshot)
#     # screenshot_cv2 = cv2.cvtColor(screenshot_cv2, cv2.COLOR_BGR2RGB)
    
#     transformed_image = transforms_1(image=screenshot_cv2)
#     transformed_image = transformed_image["image"]
    
#     with torch.no_grad():
#         prediction_1 = model_1([(transformed_image/255).to(device)])
#         pred_1 = prediction_1[0]
    
#     dieCoordinates = pred_1['boxes'][pred_1['scores'] > MIN_SCORE]
#     die_class_indexes = pred_1['labels'][pred_1['scores'] > MIN_SCORE]
    
#     return (dieCoordinates, die_class_indexes)
    

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
        
        x_move = int( (most_centered_to_enemy_x + x_screen_start) * 2/3 )
        y_move = int( (most_centered_to_enemy_y + y_screen_start) * 2/3 )
    
    return (x_move, y_move)


def cursor(x,y):
    win32api.SetCursorPos((x,y))


def left_click(x, y, time_sleep = 0):
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
    x_start = int((x_screen_start + inv_start_x)*2/3)
    y_start = int((y_screen_start + inv_start_y)*2/3)
    
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
    
    x_start = int((x_screen_start + inv_start_x)*2/3)
    y_start = int((y_screen_start + inv_start_y)*2/3)
    
    left_click(x_start, y_start, time_sleep = 0.6)
    
    win32api.keybd_event(0x26, 0,0,0)
    time.sleep(0.6)
    win32api.keybd_event(0x26, 0 ,win32con.KEYEVENTF_KEYUP ,0)
    # -------------------------------------------------------------------------
    
    
    # Runs to bank section
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
    
    enemy_coordinates_list = dieCoordinates[die_class_indexes == 4].tolist() 
    
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
        
        x_move = int( (x_click + x_screen_start) * 2/3 )
        y_move = int( (y_click + y_screen_start) * 2/3 )
    
    left_click(x_move, y_move, time_sleep = 2)
    # -------------------------------------------------------------------------
    
    
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
        
        x_move = int( (x_click + x_screen_start) * 2/3 )
        y_move = int( (y_click + y_screen_start) * 2/3 )
    
    left_click(x_move, y_move, time_sleep = 2)
    # -------------------------------------------------------------------------
    
    
    # bank deposit all inventory section
    inv_start_x = 1150
    inv_start_y = 1080
    
    x_start = int((x_screen_start + inv_start_x)*2/3)
    y_start = int((y_screen_start + inv_start_y)*2/3)
    
    left_click(x_start, y_start, time_sleep = 1)
    
    
    # # Grabs Pickaxe Section
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
    
    # enemy_coordinates_list = dieCoordinates[die_class_indexes == 10].tolist() 
    
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
        
    #     x_move = int( (x_click + x_screen_start) * 2/3 )
    #     y_move = int( (y_click + y_screen_start) * 2/3 )
    
    # left_click(x_move, y_move, time_sleep = 1)
    # # -------------------------------------------------------------------------
    
    
    # Runs back section
    # -------------------------------------------------------------------------
    inv_start_x = 4970 # Iron: 1965; Coal: 4970
    inv_start_y = 188 # Iron: 150; Coal: 188
    
    x_start = int((x_screen_start + inv_start_x)*2/3)
    y_start = int((y_screen_start + inv_start_y)*2/3)
    
    
    left_click(x_start, y_start, time_sleep = 5.5)
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
    
    x_start = int((x_screen_start + inv_start_x)*2/3)
    y_start = int((y_screen_start + inv_start_y)*2/3)
    
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
    
    x_start = int((x_screen_start + inv_start_x)*2/3)
    y_start = int((y_screen_start + inv_start_y)*2/3)
    
    left_click(x_start, y_start, 13)
    
    # Coordinates of first inventory slot relative to screen start
    inv_start_x = 1820+100
    inv_start_y = 60+5
    
    x_start = int((x_screen_start + inv_start_x)*2/3)
    y_start = int((y_screen_start + inv_start_y)*2/3)
    
    left_click(x_start, y_start, 13)
    
    left_click(x_start, y_start, 13)
    
    # Coordinates of first inventory slot relative to screen start
    inv_start_x = 1820+10
    inv_start_y = 60+5+50
    
    x_start = int((x_screen_start + inv_start_x)*2/3)
    y_start = int((y_screen_start + inv_start_y)*2/3)
    
    left_click(x_start, y_start, 13)
    
    # Coordinates of first inventory slot relative to screen start
    inv_start_x = 1820
    inv_start_y = 60+5+120
    
    x_start = int((x_screen_start + inv_start_x)*2/3)
    y_start = int((y_screen_start + inv_start_y)*2/3)
    
    left_click(x_start, y_start, 14)
    
    # Coordinates of first inventory slot relative to screen start
    inv_start_x = 1820+50
    inv_start_y = 60+5+150
    
    x_start = int((x_screen_start + inv_start_x)*2/3)
    y_start = int((y_screen_start + inv_start_y)*2/3)
    
    left_click(x_start, y_start, 13)
    # -------------------------------------------------------------------------
    
    
    # Clicks banks
    # -------------------------------------------------------------------------
    # Coordinates of first inventory slot relative to screen start
    inv_start_x = 1050
    inv_start_y = 800
    
    x_start = int((x_screen_start + inv_start_x)*2/3)
    y_start = int((y_screen_start + inv_start_y)*2/3)
    
    left_click(x_start, y_start, 2)
    # -------------------------------------------------------------------------
    
    
    # bank deposit all inventory section
    # -------------------------------------------------------------------------
    inv_start_x = 1150
    inv_start_y = 1080
    
    x_start = int((x_screen_start + inv_start_x)*2/3)
    y_start = int((y_screen_start + inv_start_y)*2/3)
    
    
    left_click(x_start, y_start, 1)
    
    
    # Runs back section
    # -------------------------------------------------------------------------
    # Coordinates of first inventory slot relative to screen start
    inv_start_x = 1820+180
    inv_start_y = 60+60
    
    x_start = int((x_screen_start + inv_start_x)*2/3)
    y_start = int((y_screen_start + inv_start_y)*2/3)
    
    left_click(x_start, y_start, 13)
    
    # Coordinates of first inventory slot relative to screen start
    inv_start_x = 1820+180
    inv_start_y = 200
    
    x_start = int((x_screen_start + inv_start_x)*2/3)
    y_start = int((y_screen_start + inv_start_y)*2/3)
    
    left_click(x_start, y_start, 13)
    
    # Coordinates of first inventory slot relative to screen start
    inv_start_x = 1820+140
    inv_start_y = 260
    
    x_start = int((x_screen_start + inv_start_x)*2/3)
    y_start = int((y_screen_start + inv_start_y)*2/3)
    
    left_click(x_start, y_start, 13)
    
    # Coordinates of first inventory slot relative to screen start
    inv_start_x = 1820+100
    inv_start_y = 260
    
    x_start = int((x_screen_start + inv_start_x)*2/3)
    y_start = int((y_screen_start + inv_start_y)*2/3)
    
    left_click(x_start, y_start, 13)
    
    left_click(x_start, y_start, 13)
    
    # Coordinates of first inventory slot relative to screen start
    inv_start_x = 1820+50
    inv_start_y = 210
    
    x_start = int((x_screen_start + inv_start_x)*2/3)
    y_start = int((y_screen_start + inv_start_y)*2/3)
    
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
    
    x_start = int((x_screen_start + inv_start_x)*2/3)
    y_start = int((y_screen_start + inv_start_y)*2/3)
    
    
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
    
    x_start = int((x_screen_start + inv_start_x)*2/3)
    y_start = int((y_screen_start + inv_start_y)*2/3)
    
    left_click(x_start, y_start, 20)
    # -------------------------------------------------------------------------
    
    
    # Clicks banks
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    
    
    # bank deposit all inventory section
    # -------------------------------------------------------------------------
    inv_start_x = 1150
    inv_start_y = 1080
    
    x_start = int((x_screen_start + inv_start_x)*2/3)
    y_start = int((y_screen_start + inv_start_y)*2/3)
    
    
    left_click(x_start, y_start, 1)
    
    
    # Runs back section
    # -------------------------------------------------------------------------
    # Coordinates of first inventory slot relative to screen start
    inv_start_x = 1820+180
    inv_start_y = 60+60
    
    x_start = int((x_screen_start + inv_start_x)*2/3)
    y_start = int((y_screen_start + inv_start_y)*2/3)
    
    left_click(x_start, y_start, 13)
    
    # Coordinates of first inventory slot relative to screen start
    inv_start_x = 1820+180
    inv_start_y = 200
    
    x_start = int((x_screen_start + inv_start_x)*2/3)
    y_start = int((y_screen_start + inv_start_y)*2/3)
    
    left_click(x_start, y_start, 13)
    
    # Coordinates of first inventory slot relative to screen start
    inv_start_x = 1820+140
    inv_start_y = 260
    
    x_start = int((x_screen_start + inv_start_x)*2/3)
    y_start = int((y_screen_start + inv_start_y)*2/3)
    
    left_click(x_start, y_start, 13)
    
    # Coordinates of first inventory slot relative to screen start
    inv_start_x = 1820+100
    inv_start_y = 260
    
    x_start = int((x_screen_start + inv_start_x)*2/3)
    y_start = int((y_screen_start + inv_start_y)*2/3)
    
    left_click(x_start, y_start, 13)
    
    left_click(x_start, y_start, 13)
    
    # Coordinates of first inventory slot relative to screen start
    inv_start_x = 1820+50
    inv_start_y = 210
    
    x_start = int((x_screen_start + inv_start_x)*2/3)
    y_start = int((y_screen_start + inv_start_y)*2/3)
    
    left_click(x_start, y_start, 13)
    # -------------------------------------------------------------------------


def mining(x_screen_start, y_screen_start, ii, stop_index):
    
    # winsound.Beep(frequency, duration)
    temp_screenshot = ImageGrab.grab(bbox =(screenshot_sizer.size[0]-1500, 
                                       500,
                                       screenshot_sizer.size[0]-350, 
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
    
    enemy_coordinates_list = dieCoordinates[die_class_indexes == 5].tolist()
    
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
        
        # x_move = int( (most_centered_to_enemy_x + x_screen_start) * 2/3 )
        # y_move = int( (most_centered_to_enemy_y + y_screen_start) * 2/3 )
        x_move = int( (x_click + x_screen_start) * 2/3 )
        y_move = int( (y_click + y_screen_start) * 2/3 )
        
        # x_move = int( (center_enemy_x_len_list[0] + x_screen_start) * 2/3 )
        # y_move = int( (center_enemy_y_len_list[0] + y_screen_start) * 2/3 )
        
        win32api.SetCursorPos((x_move, y_move))
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x_move, y_move, 0, 0)
        time.sleep(0.1)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x_move, y_move, 0, 0)
        time.sleep(0.01)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x_move, y_move, 0, 0)
        time.sleep(0.1)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x_move, y_move, 0, 0)
        time.sleep(TIME_BETWEEN_MINING)
        # time.sleep(random.randrange(1))
        return stop_index
    else:
        stop_index += 1
        if stop_index == 20:
            print("Stopping!")
            sys.exit()
        return stop_index



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
model_1 = models.detection.fasterrcnn_resnet50_fpn(pretrained=True, box_detections_per_img=500)
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
x_screen_start = screenshot_sizer.size[0]-1500
y_screen_start = 500


for i in range(1000):
    stop_index = 0
    for ii in range(28):
        stop_index = mining(x_screen_start, y_screen_start, ii, stop_index)
    
    banker()
    
    # banker_varrock()
    
    # drop_inventory()

# win32api.SetCursorPos((int((x_screen_start+1025)*2/3), int((y_screen_start+625)*2/3)))
# win32api.SetCursorPos((int((x_screen_start+950)*2/3), int((y_screen_start+700)*2/3)))




