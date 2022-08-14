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
SAVE_NAME_OD = "./Models/OSRS_Agility-0.model"
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
    # winsound.Beep(frequency, duration)
    time.sleep(time_sleep)


def fix_minimap():
    
    left_click(int(4820*2/3), int(70*2/3), time_sleep = 0.6)
    
    win32api.keybd_event(0x26, 0,0,0)
    time.sleep(0.6)
    win32api.keybd_event(0x26, 0 ,win32con.KEYEVENTF_KEYUP ,0)


def predicter(
              ):
    
    # screenshot_x1 = screenshot_sizer.size[0]-2100, 
    # screenshot_y1 = 0, 
    # screenshot_x2 = screenshot_sizer.size[0], 
    # screenshot_y2 = screenshot_sizer.size[1]
    
    screenshot_sizer = ImageGrab.grab()
    
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
    
    return (dieCoordinates, die_class_indexes)
    

def coord_to_move_to(dieCoordinates, die_class_indexes, interested_index=0):
    
    # checks if needs to reset to start location
    
    if (len(dieCoordinates[die_class_indexes == 10]) > 0
        and len(dieCoordinates[die_class_indexes == 1]) == 0): # If "Minimap-Start_Location" available
        enemy_coordinates_list = dieCoordinates[die_class_indexes == 10].tolist()
        needs_repeat = True
    elif (len(dieCoordinates[die_class_indexes == 9]) > 0
        and len(dieCoordinates[die_class_indexes == 1]) == 0): # If "Minimap-Lost_Location" available
        enemy_coordinates_list = dieCoordinates[die_class_indexes == 9].tolist()
        needs_repeat = True
    else:
        if len(dieCoordinates[die_class_indexes == 8]) > 0: # If Mark of Grace found on ground - CHANGE NUMBER 1
            if (dieCoordinates[die_class_indexes == 8][0][0] > int(2100*.15)
                and dieCoordinates[die_class_indexes == 8][0][2] < int(2100*.85)
                and dieCoordinates[die_class_indexes == 8][0][1] > int(screenshot_sizer.size[1]*.15)
                and dieCoordinates[die_class_indexes == 8][0][3] < int(screenshot_sizer.size[1]*.85)
                ): # box within 25%-50% of image
                enemy_coordinates_list = dieCoordinates[die_class_indexes == 8].tolist()
                needs_repeat = True
            else:
                enemy_coordinates_list = dieCoordinates[die_class_indexes == interested_index].tolist()
                needs_repeat = False
        else:
            enemy_coordinates_list = dieCoordinates[die_class_indexes == interested_index].tolist()
            needs_repeat = False
    
    
    center_enemy_x = int(enemy_coordinates_list[0][0]
                        +(enemy_coordinates_list[0][2]-enemy_coordinates_list[0][0])/2
                        )
    center_enemy_y = int(enemy_coordinates_list[0][1]
                        +(enemy_coordinates_list[0][3]-enemy_coordinates_list[0][1])/2
                        )
        
    x_move = int( (center_enemy_x + x_screen_start) * 2/3 )
    y_move = int( (center_enemy_y + y_screen_start) * 2/3 )
    
    return (x_move, y_move, needs_repeat)


def agility_trainer():
    # Sets a part for future
    needs_repeat = True
    
    # Fixes minimap
    # -------------------------------------------------------------------------
    fix_minimap()
    # -------------------------------------------------------------------------
    
    # Starts location 1
    # -------------------------------------------------------------------------
    step_index = 0
    while needs_repeat:
        dieCoordinates, die_class_indexes = predicter()
        x_move, y_move, needs_repeat = coord_to_move_to(dieCoordinates, 
                                                        die_class_indexes, 
                                                        interested_index=1)
        left_click(x_move, y_move, time_sleep = 7)
        if needs_repeat:
            step_index += 1
        if step_index > 5:
            break
    if step_index > 5:
        return
    needs_repeat = True
    # -------------------------------------------------------------------------
    
    # Starts location 2
    # -------------------------------------------------------------------------
    step_index = 0
    while needs_repeat:
        dieCoordinates, die_class_indexes = predicter()
        x_move, y_move, needs_repeat = coord_to_move_to(dieCoordinates, 
                                                        die_class_indexes, 
                                                        interested_index=2)
        left_click(x_move, y_move, time_sleep = 9)
        if needs_repeat:
            step_index += 1
        if step_index > 5:
            break
    if step_index > 5:
        return
    needs_repeat = True
    # -------------------------------------------------------------------------
    
    # Starts location 3
    # -------------------------------------------------------------------------
    step_index = 0
    while needs_repeat:
        dieCoordinates, die_class_indexes = predicter()
        x_move, y_move, needs_repeat = coord_to_move_to(dieCoordinates, 
                                                        die_class_indexes, 
                                                        interested_index=3)
        left_click(x_move, y_move, time_sleep = 10)
        if needs_repeat:
            step_index += 1
        if step_index > 5:
            break
    if step_index > 5:
        return
    needs_repeat = True
    # -------------------------------------------------------------------------
    
    # Starts location 4
    # -------------------------------------------------------------------------
    step_index = 0
    while needs_repeat:
        dieCoordinates, die_class_indexes = predicter()
        x_move, y_move, needs_repeat = coord_to_move_to(dieCoordinates, 
                                                        die_class_indexes, 
                                                        interested_index=4)
        left_click(x_move, y_move, time_sleep = 7)
        if needs_repeat:
            step_index += 1
        if step_index > 5:
            break
    if step_index > 5:
        return
    needs_repeat = True
    # -------------------------------------------------------------------------
    
    # Starts location 5
    # -------------------------------------------------------------------------
    step_index = 0
    while needs_repeat:
        dieCoordinates, die_class_indexes = predicter()
        x_move, y_move, needs_repeat = coord_to_move_to(dieCoordinates, 
                                                        die_class_indexes, 
                                                        interested_index=5)
        left_click(x_move, y_move, time_sleep = 5)
        if needs_repeat:
            step_index += 1
        if step_index > 5:
            break
    if step_index > 5:
        return
    needs_repeat = True
    # -------------------------------------------------------------------------
    
    # Starts location 6
    # -------------------------------------------------------------------------
    step_index = 0
    while needs_repeat:
        dieCoordinates, die_class_indexes = predicter()
        x_move, y_move, needs_repeat = coord_to_move_to(dieCoordinates, 
                                                        die_class_indexes, 
                                                        interested_index=6)
        left_click(x_move, y_move, time_sleep = 5)
        if needs_repeat:
            step_index += 1
        if step_index > 5:
            break
    if step_index > 5:
        return
    needs_repeat = True
    # -------------------------------------------------------------------------
    
    # Starts location 7
    # -------------------------------------------------------------------------
    step_index = 0
    while needs_repeat:
        dieCoordinates, die_class_indexes = predicter()
        x_move, y_move, needs_repeat = coord_to_move_to(dieCoordinates, 
                                                        die_class_indexes, 
                                                        interested_index=7)
        left_click(x_move, y_move, time_sleep = 5)
        if needs_repeat:
            step_index += 1
        if step_index > 5:
            break
    if step_index > 6:
        return
    needs_repeat = True
    # -------------------------------------------------------------------------



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
x_screen_start = screenshot_sizer.size[0]-2100
y_screen_start = 0


for i in range(1000):
    agility_trainer()




