import time
from PIL import ImageGrab
import winsound



# Windows beep settings
frequency = 700  # Set Frequency To 2500 Hertz
duration = 80  # Set Duration To 1000 ms == 1 second


time.sleep(1)

screenshot_sizer = ImageGrab.grab()

winsound.Beep(frequency, duration)

for i in range(5):
    screenshot = ImageGrab.grab(bbox =(screenshot_sizer.size[0]-2100, 
                                       0,
                                       screenshot_sizer.size[0], 
                                       screenshot_sizer.size[1]
                                       )
                                )
    
    screenshot.save('./Images/Screenshots/image-{}.jpg'.format(i))
    
    time.sleep(0.5)
    winsound.Beep(frequency, 40)
    
winsound.Beep(frequency-100, duration)