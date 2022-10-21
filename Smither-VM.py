import re
# remove arnings (optional)
import warnings
warnings.filterwarnings("ignore")
import time
import win32api, win32con
import random
from pynput.keyboard import Key, Controller
from datetime import datetime


# User parameters
SAVE_NAME_OD = "./Models/OSRS_Agility-0.model"
DATASET_PATH = "./Training_Data/" + SAVE_NAME_OD.split("./Models/",1)[1].split("-",1)[0] +"/"
IMAGE_SIZE              = int(re.findall(r'\d+', SAVE_NAME_OD)[-1] ) # Row and column number 
MIN_SCORE               = 0.7
RESOLUTION_X            = 1920
RESOLUTION_Y            = 976
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


def cursor(x,y):
    x = round(x * RESOLUTION_X)
    y = round(y * RESOLUTION_Y)
    win32api.SetCursorPos((x,y))


def left_click(x, y, time_sleep = 0, should_rand_click = True):
    x = round(x * RESOLUTION_X)
    y = round(y * RESOLUTION_Y)
    
    if should_rand_click:
        rand_scaler_x = round(RESOLUTION_X * 0.002)
        rand_scaler_y = round(RESOLUTION_Y * 0.002)
        x = x + random.randint(-rand_scaler_x, rand_scaler_x)
        y = y + random.randint(-rand_scaler_y, rand_scaler_y)
    
    win32api.SetCursorPos((x,y))
    time.sleep(0.1)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)
    # winsound.Beep(frequency, duration)
    time.sleep(time_sleep)
    time.sleep(random.randrange(1))


def fix_minimap():
    
    left_click(0.895, 0.05, time_sleep = 0.6)
    
    win32api.keybd_event(0x26, 0,0,0)
    time.sleep(0.6)
    win32api.keybd_event(0x26, 0 ,win32con.KEYEVENTF_KEYUP, 0)


def smelter():
    # Clicks Anvil
    left_click(0.495, 0.580, time_sleep=1.5)
    
    # Clicks item to smith
    x = 0.49-0.055*0
    y = 0.41-0.055*2
    left_click(x, y, time_sleep=80)


def banker():
    fix_minimap()
    
    # Runs to bank
    left_click(0.931, 0.072, time_sleep=5, should_rand_click = False)
    
    # Clicks bank
    left_click(0.54, 0.45, time_sleep=1)
    
    # Deposits all
    left_click(0.92, 0.67, time_sleep=0.5)
    
    # Withdraw bars
    left_click(0.35, 0.495, time_sleep=0.5)
    
    # Runs to anvil
    left_click(0.945, 0.147, time_sleep=5, should_rand_click = False)


def date_time():
    # Gets date and time
    now = datetime.now()
    hour = int(now.strftime("%H"))
    # Checks if appropriate time to bot
    if (hour >= 7 and hour <= 24) or hour <= 2:
        should_continue = True
    else:
        should_continue = False
    return should_continue


def logout():
    # Clicks X Button
    left_click(0.97, 0.03, time_sleep = 1)
    
    # Clicks Logout Button
    left_click(0.97, 0.90, time_sleep = 1)
    left_click(0.94, 0.89, time_sleep = 1)


def login():
    # Click Existing Log In
    left_click(0.54, 0.33, time_sleep = 1)
    
    # Types password in
    typeThis("1Nomad2")
    
    # Sleeps
    time.sleep(10)
    
    # Click to Play
    left_click(0.52, 0.35, time_sleep = 5)
    
    # Clicks Inventory
    left_click(0.82, 0.943, time_sleep = 1)



def logger():
    logout()
    
    # Sleeps
    time.sleep(10 + random.randrange(500))
    
    login()
    



# Main()
keyboard = Controller()

should_continue = date_time()

altTab()
time.sleep(1)

while True:
    if not should_continue:
        time.sleep(60*30)
        should_continue = date_time()
        
        if should_continue:
            login()
        
    
    while should_continue:
    
        for i in range(int(1620/27)):
            smelter()
            
            banker()
        
        logger()
        
        should_continue = date_time()



