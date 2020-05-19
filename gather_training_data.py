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
MAX_SESSIONS = 20 # needs to match value in train_model.py - todo move to proper config file

# init
frames_processed = 0


if __name__ == "__main__":
    print("Running...")
    
    # read training data
    print("Checking for existing training data...")
    for session_number in range(1, MAX_SESSIONS):
        if session_number >= MAX_SESSIONS:
            print(f"Warning: You might be exceeding {MAX_SESSIONS} training sessions. " +
                  f"Raise MAX_SESSIONS if you want to train on more sessions.")
        TRAINING_SESSION_ID = "session_" + str(session_number)
        try:
            with open(f'training_data/training_data_{TRAINING_SESSION_ID}.npy', 'rb') as f:
                pass
        except FileNotFoundError:
            num_sessions_loaded = session_number - 1
            training_session_id = "session_" + str(num_sessions_loaded + 1)
            print(f"Found {num_sessions_loaded} files of training data")
            break

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
    filename = f'training_data/training_data_{training_session_id}.npy'
    np.save(filename, screen_captures)
    print(f"Saved {len(screen_captures)} screen captures of shape {screen_captures[0].shape} to {filename}")
        
    # save training labels
    np.save(f'training_data/training_labels_{training_session_id}.npy', labels)
    print(f"Saved {len(labels)} training labels of shape {labels[0].shape} to {filename}")

    print("completed successfully")
