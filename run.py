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
from protodriver import utils


#config
COUNT_DOWN = True
MAX_FRAMES = 40 # none for infinite runtime, roughly 10 fps for training and 1.5 fps for running
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

# init
frames_processed = 0


if __name__ == "__main__":
    print("running")

    # countdown
    if COUNT_DOWN:
        for count in range(3, 0, -1):
            print(count)
            time.sleep(0.5)

    # init
    user_exit = False
    user_pause = True
    last_time = time.time()
    if MAX_FRAMES is None:
        MAX_FRAMES = int("inf")
    model = keras.models.load_model('models/basic_cnn')
    model.summary()

    # game loop
    while frames_processed < MAX_FRAMES and not user_exit:
        frame_number = frames_processed  # todo - make name of frame independent of how many processed
        
        # grab screen
        screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))

        # process image and display resulting image
        processed_screen = utils.process_image(screen)
        #screen_captures.append(processed_screen)
        cv2.imshow('window', processed_screen)

        user_input = utils.get_user_input()
        #if user_input[5]:  # c pressed
        #    user_exit = True
        #elif user_input[4]:  # space pressed
        #    user_pause = True

        # need to rethink pausing - maybe one key to start/unpause and another pause
        # this imght be a good time to reorganize project structure
            
        # get model prediction 
        model_input = np.array(processed_screen).reshape((1, 75, 100, 3))
        prediction = model.predict(model_input)[0]
        prediction_str = " ".join([f"{p:2.2}" for p in prediction])
        print(prediction_str)

        # send input
        utils.send_input(prediction)
        
        # some stuff to get opencv not to crash
        if(cv2.waitKey(25) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break

        # display framerate
        fps = 1 / (time.time() - last_time)
        last_time = time.time()
        frames_processed = frames_processed + 1
        print(f"Framerate: {fps:4.4} fps, ({frames_processed} / {MAX_FRAMES}) frames processed")    
        
    # feet off the pedals!
    pydirectinput.keyUp('w')
    pydirectinput.keyUp('s')
    
    print("completed successfully")
