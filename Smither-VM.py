import re
# remove arnings (optional)
import warnings
warnings.filterwarnings("ignore")
import time
import win32api, win32con
import random
from pynput.keyboard import Key, Controller


# User parameters
SAVE_NAME_OD = "./Models/OSRS_Agility-0.model"
DATASET_PATH = "./Training_Data/" + SAVE_NAME_OD.split("./Models/",1)[1].split("-",1)[0] +"/"
IMAGE_SIZE              = int(re.findall(r'\d+', SAVE_NAME_OD)[-1] ) # Row and column number 
MIN_SCORE               = 0.7
TIME_BETWEEN_MINING     = 4 # Set 2.0 default for one pick iron
# User Parameters/Constants to Set
XML_DIR = "./XML_Files/"
SLEEP_TIME_BETWEEN_TYPETHIS = 0.01
SLEEP_TIME_BETWEEN_LETTER = 0.01


# Presses Alt + Tab
def altTab():
    time.sleep(SLEEP_TIME_BETWEEN_TYPETHIS)
    with keyboard.pressed(Key.alt):
        keyboard.press(Key.tab)
        keyboard.release(Key.tab)


# Type's in string
def typeThis(toType):
    time.sleep(SLEEP_TIME_BETWEEN_TYPETHIS)
    for letter in toType:
        keyboard.type(letter)
        time.sleep(SLEEP_TIME_BETWEEN_LETTER)
    time.sleep(SLEEP_TIME_BETWEEN_TYPETHIS)
    keyboard.press(Key.enter)
    keyboard.release(Key.enter)
    time.sleep(SLEEP_TIME_BETWEEN_TYPETHIS)


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
    
    left_click(964, 37, time_sleep = 0.6, should_scaler=False)
    
    win32api.keybd_event(0x26, 0,0,0)
    time.sleep(0.6)
    win32api.keybd_event(0x26, 0 ,win32con.KEYEVENTF_KEYUP, 0)


def smelter():
    # Clicks Anvil
    left_click(813, 435, time_sleep=1.5)
    
    # Clicks item to smith
    x = 808-24*0
    y = 315-43*2
    left_click(x, y, time_sleep=80)


def banker():
    fix_minimap()
    
    # Runs to bank
    left_click(984, 60, time_sleep=5, should_rand_click = False)
    
    # Clicks bank
    left_click(834, 347, time_sleep=1)
    
    # Deposits all
    left_click(974, 940, time_sleep=0.5)
    
    # Withdraw bars
    left_click(727, 501, time_sleep=0.5)
    
    # Runs to anvil
    left_click(991+0, 117+0, time_sleep=5, should_rand_click = False)



# Main()
keyboard = Controller()

altTab()
time.sleep(1)

# resolution: VM: 1024 x 768; PC: 5120 x 1440

for i in range(int(4320/27)):
    smelter()
    
    banker()




