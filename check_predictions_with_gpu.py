import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui
import pydirectinput
import keyboard
import tensorflow as tf
from tensorflow import keras
import os


#config
#GAME_TYPE = "Run" # "Train", "Run"  todo - move to file argument
COUNT_DOWN = True
MAX_FRAMES = 10
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
                          #bot left  mid left   top left    top right   mid right   bot right
ROI_VERTICES = [np.array([[10, 500], [10, 250], [399, 200], [401, 200], [800, 250], [800, 500]])]
LINE_COLOR = [255, 255, 255] # white
LINE_WIDTH = 3 # pixels
#tf.config.experimental.set_per_process_memory_fraction(0.75)    #diff
#tf.config.experimental.set_memory_growth("0", True)    #diff
# init
frames_processed = 0

def _region_of_interest(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(image, mask)


def _draw_lines(image, lines):
    """draws lines on image"""
    if lines is None:
        return
    
    for line in lines:
        #if line is not None and len(line) == 4:
        cv2.line(image, (line[0][0], line[0][1]), (line[0][2], line[0][3]), LINE_COLOR, LINE_WIDTH)


def process_image(original_image):
    processed_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_image = _region_of_interest(processed_image, ROI_VERTICES)

    # edge detection
    processed_image = cv2.Canny(processed_image, threshold1=100, threshold2=200, edges=None, apertureSize=3)

    lines = cv2.HoughLinesP(image=processed_image, rho=1, theta=np.pi/180,
                            threshold=60, lines=None, minLineLength=100, maxLineGap=5)
    _draw_lines(processed_image, lines)

    processed_image = cv2.pyrDown(src=processed_image, dstsize=(400,300))

    return processed_image

















def send_input(prediction):     # diff
    print_str = ''
    if prediction[0] > 0.5:
        print_str = print_str + 'w'

    if prediction[1] > 0.5:
        print_str = print_str + 'a'
        
    if prediction[2] > 0.5:
        print_str = print_str + 's'
        
    if prediction[3] > 0.5:
        print_str = print_str + 'd'

    print(print_str)







if __name__ == "__main__":
    print("running")

    # countdown
    if COUNT_DOWN:
        for count in range(3, 0, -1):
            print(count)
            time.sleep(0.5)

    # init

        # diff

        
    last_time = time.time()
    if MAX_FRAMES is None:
        MAX_FRAMES = int("inf")
    screen_captures = []
            # diff    
    model = keras.models.load_model('models/basic_cnn')
    model.summary()

    # game loop
    while frames_processed < MAX_FRAMES:
        frame_number = frames_processed  # todo - make name of frame independent of how many processed
        
        # grab screen
        screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))


            # diff (user input)





        


        # process image and display resulting image
        processed_screen = process_image(screen)
        screen_captures.append(processed_screen)
        cv2.imshow('window', processed_screen)


            
        # get model prediction 
        model_input = np.expand_dims(np.array(processed_screen), -1).reshape((1, 300, 400, 1))
        #print(model_input.shape)
        prediction = model.predict(model_input)[0]
        print(prediction)


        # send input
        send_input(prediction)
        
        # some stuff to get opencv not to crash
        if(cv2.waitKey(25) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break

        # display framerate
        fps = 1 / (time.time() - last_time)
        last_time = time.time()
        frames_processed = frames_processed + 1
        print(f"Framerate: {fps:4.4} fps, ({frames_processed} / {MAX_FRAMES}) frames processed")    


    print("completed successfully")
