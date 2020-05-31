import numpy as np
from PIL import ImageGrab
import cv2
import time
from collections import deque
import pyautogui
import pydirectinput
import os
from protodriver import utils

# config
COUNT_DOWN = True
MAX_FRAMES = 500  # none for infinite runtime, roughly 5-10fps
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class baseline_model():
    """
    Baseline model that simply chooses steering direction and keeps a minimum throttle input
    """
    def __init__(self):
        self.throttle_fraction = 0.11
        self._num_throttle_to_remember = 10
        self._last_throttles = deque(np.zeros(self._num_throttle_to_remember),
                                     self._num_throttle_to_remember)
        self.MAX_CURVATURE_TO_TURN = 150

    def act(self, unprocessed_screen, show_processed_screen=True):
        # process image
        processed_screen, curvature, direction = utils.process_image_for_baseline(unprocessed_screen)
        if show_processed_screen:
            cv2.imshow('window', processed_screen)

        # get prediction
        steering = self.get_steering_input(curvature, direction)
        throttle = self.get_throttle_input()
        self._last_throttles.append(throttle)

        # send input
        send_steering_input(steering)
        send_throttle_input(throttle)
        return steering, throttle, processed_screen

    def get_steering_input(self, curvature, direction):
        if curvature <= self.MAX_CURVATURE_TO_TURN:
            return direction
        else:
            return 'straight'

    def get_throttle_input(self):
        return np.mean(self._last_throttles) <= self.throttle_fraction


def send_throttle_input(throttle_on):
    if throttle_on:
        pydirectinput.keyDown('w', _pause=False)
    else:
        pydirectinput.keyUp('w', _pause=False)


def send_steering_input(prediction):
    """

    Parameters
    ----------
    prediction : string 'left', 'right', or 'straight'

    Returns
    -------
    Doesn't return anything, just sends input
    """
    if prediction == 'left':
        pydirectinput.keyDown('a', _pause=False)
        pydirectinput.keyUp('d', _pause=False)
    elif prediction == 'right':
        pydirectinput.keyDown('d', _pause=False)
        pydirectinput.keyUp('a', _pause=False)
    elif prediction == 'straight':
        pydirectinput.keyUp('d', _pause=False)
        pydirectinput.keyUp('a', _pause=False)


if __name__ == "__main__":
    print("running - press space to exit")

    # countdown
    if COUNT_DOWN:
        for count in range(3, 0, -1):
            print(count)
            time.sleep(0.5)

    # init
    frames_processed = 0
    user_exit = False
    last_time = time.time()
    if MAX_FRAMES is None:
        MAX_FRAMES = int("inf")
    model = baseline_model()

    # game loop
    while frames_processed < MAX_FRAMES and not user_exit:
        frame_number = frames_processed  # todo - make name of frame independent of how many processed

        # grab screen
        screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))

        user_input = utils.get_user_input()
        if user_input[4]:  # space pressed
            user_exit = True

        # get model prediction
        steering, throttle, _ = model.act(screen)
        print('Steering:', steering, 'Throttle', throttle)

        # some stuff to get opencv not to crash
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        # display framerate
        fps = 1 / (time.time() - last_time)
        last_time = time.time()
        frames_processed = frames_processed + 1
        print(f"Framerate: {fps:4.4} fps, ({frames_processed} / {MAX_FRAMES}) frames processed")

        # feet off the pedals!
    pydirectinput.keyUp('w', _pause=False)
    pydirectinput.keyUp('s', _pause=False)

    print("completed successfully")
