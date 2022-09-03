import pyautogui
import time

while True:
    time.sleep(2)
    test = pyautogui.position()
    print(test)