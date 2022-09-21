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


def cursor(x,y):
    win32api.SetCursorPos((x,y))


def left_click(x, y, time_sleep = 0, 
               should_rand_click = True, should_scaler = False):
    if should_rand_click:
        x = x + random.randint(-4, 4)
        y = y + random.randint(-4, 4)
    if should_scaler:
        x = int(x*2/3)
        y = int(y*2/3)
    win32api.SetCursorPos((x,y))
    time.sleep(0.1)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)
    # winsound.Beep(frequency, duration)
    time.sleep(time_sleep)
    time.sleep(random.randrange(1))


def fix_minimap():
    
    left_click(4820, 70, time_sleep = 0.6, should_rand_click = False, should_scaler=True)
    
    win32api.keybd_event(0x26, 0,0,0)
    time.sleep(0.6)
    win32api.keybd_event(0x26, 0 ,win32con.KEYEVENTF_KEYUP, 0)


def smelter():
    # Clicks Furnace
    left_click(2800, 460, time_sleep=1.5)
    
    # Clicks Steel
    # steel 28 sec; iron 85 sec
    # steel 3440, 1245; iron 3275, 1245
    left_click(3275, 1245, time_sleep=85, should_scaler=True) 


def banker():
    fix_minimap()
    
    # Runs to bank
    left_click(3248, 131, time_sleep=7, should_rand_click = False)
    
    # Clicks bank
    left_click(2715, 540, time_sleep=1.5)
    
    # Deposits all
    left_click(2782, 718, time_sleep=0.5)
    
    # Withdraws Iron Ore
    left_click(4064, 667, time_sleep=0.5, should_scaler=True)
    
    # Withdraws Coal Ore
    # left_click(4142, 667, time_sleep=0.5, should_scaler=True)
    
    # # Withdraws Coal Ore
    # left_click(4142, 667, time_sleep=0.5, should_scaler=True)
    
    # Runs to smelter
    left_click(3338, 90, time_sleep=7, should_rand_click = False)



# Main()


for i in range(1000):
    smelter()
    
    banker()




