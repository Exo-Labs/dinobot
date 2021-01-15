from mss.windows import MSS as mss
import pyautogui as pg
import numpy as np
import time
import cv2

box = {'top': 510, 'left': 240, 'width': 21, 'height': 20}

def image_processing(init_img):
    processed_image = cv2.cvtColor(init_img, cv2.COLOR_BGR2GRAY)
    processed_image = cv2.Canny(processed_image, threshold1 = 200, threshold2 = 200)
    return processed_image

def action(mean):
    if not mean == float(0):
            pg.press('up')
    
def vision():
    sct = mss()
    last_time = time.time()
    
    while (True):
        img = sct.grab(box)
        print('loop processed for {} seconds'.format(time.time() - last_time))
        last_time = time.time()
        img = np.array(img)
        processed_image = image_processing(img)
        mean = np.mean(processed_image)
        print('mean = ', mean)

        action(mean)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
vision()