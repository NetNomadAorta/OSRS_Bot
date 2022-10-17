import re
# remove arnings (optional)
import warnings
warnings.filterwarnings("ignore")
import time
import win32api, win32con
import random


# User parameters
SAVE_NAME_OD = "./Models/OSRS_Agility-0.model"
DATASET_PATH = "./Training_Data/" + SAVE_NAME_OD.split("./Models/",1)[1].split("-",1)[0] +"/"
IMAGE_SIZE              = int(re.findall(r'\d+', SAVE_NAME_OD)[-1] ) # Row and column number 
MIN_SCORE               = 0.7
TIME_BETWEEN_MINING     = 4 # Set 2.0 default for one pick iron


def cursor(x,y, should_scaler = True):
    if should_scaler:
        x = int(x*2/3)
        y = int(y*2/3)
    win32api.SetCursorPos((x,y))


def left_click(x, y, time_sleep = 0, should_scaler = True, 
               should_rand_click = True):
    if should_scaler:
        x = int(x*2/3)
        y = int(y*2/3)
    if should_rand_click:
        x = x + random.randint(-5, 5)
        y = y + random.randint(-6, 6)
    win32api.SetCursorPos((x,y))
    time.sleep(0.1)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)
    # winsound.Beep(frequency, duration)
    time.sleep(time_sleep)
    time.sleep(random.randrange(1))


def fix_minimap():
    
    left_click(4820, 70, time_sleep = 0.6, should_scaler=False)
    
    win32api.keybd_event(0x26, 0,0,0)
    time.sleep(0.6)
    win32api.keybd_event(0x26, 0 ,win32con.KEYEVENTF_KEYUP, 0)


def smelter():
    # Clicks Anvil
    left_click(4065, 815, time_sleep=1.5)
    
    # Clicks item to smith
    x = 4040-120*0
    y = 590-80*2
    left_click(x, y, time_sleep=80)


def banker():
    fix_minimap()
    
    # Runs to bank
    left_click(4921, 112, time_sleep=5, should_rand_click = False)
    
    # Clicks bank
    left_click(4170, 650, time_sleep=1)
    
    # Deposits all
    left_click(4870, 940, time_sleep=0.5)
    
    # Withdraw bars
    left_click(3635, 725, time_sleep=0.5)
    
    # Runs to anvil
    left_click(4955+0, 220+0, time_sleep=5, should_rand_click = False)



# Main()


for i in range(int(4320/27)):
    smelter()
    
    banker()




