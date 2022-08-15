import re
# remove arnings (optional)
import warnings
warnings.filterwarnings("ignore")
import time
import win32api, win32con
    

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


def smelter():
    # Clicks Furnace
    left_click(2800, 460, time_sleep=1.5)
    
    # Clicks Iron
    left_click(2185, 832, time_sleep=85)


def banker():
    fix_minimap()
    
    # Runs to bank
    left_click(3248, 131, time_sleep=6)
    
    # Clicks bank
    left_click(2715, 540, time_sleep=1.5)
    
    # Deposits all
    left_click(2782, 718, time_sleep=1)
    
    # Withdraws Iron Ore
    left_click(2714, 373, time_sleep=1)
    
    # Runs to smelter
    left_click(3338, 90, time_sleep=6)



# Main()


for i in range(1000):
    smelter()
    
    banker()




