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


# User parameters
SAVE_NAME_OD = "./Models/OSRS_Mining-0.model"
DATASET_PATH = "./Training_Data/" + SAVE_NAME_OD.split("./Models/",1)[1].split("-",1)[0] +"/"
IMAGE_SIZE              = int(re.findall(r'\d+', SAVE_NAME_OD)[-1] ) # Row and column number 
MIN_SCORE               = 0.7


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


def banker():
    # Takes screenshot to check size of screen
    screenshot_sizer = ImageGrab.grab()

    # Finds window size and where coordinates starts and ends in window
    x_screen_start = screenshot_sizer.size[0]-2100
    y_screen_start = 0
    
    # Runs to bank section
    # Coordinates of first inventory slot relative to screen start
    inv_start_x = 1820+50
    inv_start_y = 175+10
    
    x_start = int((x_screen_start + inv_start_x)*2/3)
    y_start = int((y_screen_start + inv_start_y)*2/3)
    
    win32api.SetCursorPos((x_start, y_start))
    time.sleep(0.1)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x_start, y_start, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x_start, y_start, 0, 0)
    time.sleep(9)
    
    
    # bank click section
    inv_start_x = 950
    inv_start_y = 700
    
    x_start = int((x_screen_start + inv_start_x)*2/3)
    y_start = int((y_screen_start + inv_start_y)*2/3)
    
    win32api.SetCursorPos((x_start, y_start))
    time.sleep(0.1)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x_start, y_start, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x_start, y_start, 0, 0)
    time.sleep(1)
    
    
    # bank deposit inventory section
    inv_start_x = 1800
    inv_start_y = 940
    
    x_start = int((x_screen_start + inv_start_x)*2/3)
    y_start = int((y_screen_start + inv_start_y)*2/3)
    
    win32api.SetCursorPos((x_start, y_start))
    time.sleep(0.1)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x_start, y_start, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x_start, y_start, 0, 0)
    time.sleep(1)
    
    
    # Runs back section
    inv_start_x = 2020-55
    inv_start_y = 160-10
    
    x_start = int((x_screen_start + inv_start_x)*2/3)
    y_start = int((y_screen_start + inv_start_y)*2/3)
    
    win32api.SetCursorPos((x_start, y_start))
    time.sleep(0.1)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x_start, y_start, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x_start, y_start, 0, 0)
    time.sleep(9)


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
    # BELOW SHOWS SCORES - COMMENT OUT IF NEEDED
    die_scores = pred_1['scores'][pred_1['scores'] > MIN_SCORE]
    
    enemy_coordinates_list = dieCoordinates[die_class_indexes > 0].tolist() # SHOULD "== 1].tolist()" FOR ENEMY
    
    die_class_indexes = die_class_indexes.tolist()
    # BELOW SHOWS SCORES - COMMENT OUT IF NEEDED
    die_scores = die_scores.tolist()
    
    # predicted_image = draw_bounding_boxes(transformed_image,
    #     boxes = dieCoordinates,
    #     # labels = [classes_1[i] for i in die_class_indexes], 
    #     # labels = [str(round(i,2)) for i in die_scores], # SHOWS SCORE IN LABEL
    #     width = 2,
    #     colors = "red"
    #     )
    # save_image((predicted_image/255), './Images/Screenshots/image-{}.jpg'.format(ii))
    
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
        
        # x_move = int( (center_enemy_x_len_list[0] + x_screen_start) * 2/3 )
        # y_move = int( (center_enemy_y_len_list[0] + y_screen_start) * 2/3 )
        
        time_set = 1.9
        
        win32api.SetCursorPos((x_move, y_move))
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x_move, y_move, 0, 0)
        time.sleep(0.1)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x_move, y_move, 0, 0)
        time.sleep(0.01)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x_move, y_move, 0, 0)
        time.sleep(0.1)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x_move, y_move, 0, 0)
        time.sleep(time_set)
        # time.sleep(random.randrange(1))
        return stop_index
    else:
        stop_index += 1
        if stop_index == 15:
            sys.exit()
        return stop_index
    
    
    # x_temp_1 = int((x_screen_start+1025)*2/3)
    # y_temp_1 = int((y_screen_start+625)*2/3)
    # x_temp_2 = int((x_screen_start+950)*2/3)
    # y_temp_2 = int((y_screen_start+700)*2/3)
    
    # time_set = 2
    
    # for i in range(9):
    #     win32api.SetCursorPos((x_temp_1+10, y_temp_1))
    #     win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x_temp_1, y_temp_1, 0, 0)
    #     time.sleep(0.1)
    #     win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x_temp_1, y_temp_1, 0, 0)
    #     time.sleep(0.01)
    #     win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x_temp_1, y_temp_1, 0, 0)
    #     time.sleep(0.1)
    #     win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x_temp_1, y_temp_1, 0, 0)
    #     time.sleep(time_set)
    #     time.sleep(random.randrange(1))
        
    #     win32api.SetCursorPos((x_temp_2, y_temp_2))
    #     win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x_temp_2, y_temp_2, 0, 0)
    #     time.sleep(0.1)
    #     win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x_temp_2, y_temp_2, 0, 0)
    #     time.sleep(0.01)
    #     win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x_temp_2, y_temp_2, 0, 0)
    #     time.sleep(0.1)
    #     win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x_temp_2, y_temp_2, 0, 0)
    #     time.sleep(time_set)
    #     time.sleep(random.randrange(1))
        
    #     win32api.SetCursorPos((x_temp_1+10, y_temp_1+120))
    #     win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x_temp_2, y_temp_2, 0, 0)
    #     time.sleep(0.1)
    #     win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x_temp_2, y_temp_2, 0, 0)
    #     time.sleep(0.01)
    #     win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x_temp_2, y_temp_2, 0, 0)
    #     time.sleep(0.1)
    #     win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x_temp_2, y_temp_2, 0, 0)
    #     time.sleep(time_set)
    #     time.sleep(random.randrange(1))



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


for i in range(100):
    stop_index = 0
    for ii in range(26):
        stop_index = mining(x_screen_start, y_screen_start, ii, stop_index)
    
    banker()
    
    # drop_inventory()

# win32api.SetCursorPos((int((x_screen_start+1025)*2/3), int((y_screen_start+625)*2/3)))
# win32api.SetCursorPos((int((x_screen_start+950)*2/3), int((y_screen_start+700)*2/3)))




