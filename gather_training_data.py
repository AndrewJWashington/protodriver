import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui
import pydirectinput
import keyboard
from protodriver import utils


#config
COUNT_DOWN = True
MAX_FRAMES = 1000 # none for infinite runtime, roughly 10 fps for training and 1.5 fps for running
TRAINING_SESSION_ID = "session_5"

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
    screen_captures = []
    labels = []

    # game loop
    while frames_processed < MAX_FRAMES and not user_exit:
        frame_number = frames_processed  # todo - make name of frame independent of how many processed
        
        # grab screen
        screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))

        # get user input, then store as labelled data
        user_input = utils.get_user_input()
        labels.append(user_input[:4]) # don't need to record space
        
        if user_input[4]:  # space pressed
            user_exit = True

        # process image and display resulting image
        processed_screen = utils.process_image(screen)
        screen_captures.append(processed_screen)
        cv2.imshow('window', processed_screen)

        # some stuff to get opencv not to crash
        if(cv2.waitKey(25) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break

        # display framerate
        fps = 1 / (time.time() - last_time)
        last_time = time.time()
        frames_processed = frames_processed + 1
        print(f"Framerate: {fps:4.4} fps, ({frames_processed} / {MAX_FRAMES}) frames processed")    
    
    # save training input
    filename = f'training_data/training_data_{TRAINING_SESSION_ID}.npy'
    np.save(filename, screen_captures)
    print(f"Saved {len(screen_captures)} screen captures of shape {screen_captures[0].shape} to {filename}")
        
    # save training labels
    np.save(f'training_data/training_labels_{TRAINING_SESSION_ID}.npy', labels)
    print(f"Saved {len(labels)} training labels of shape {labels[0].shape} to {filename}")

    print("completed successfully")
