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

# config
ROI_VERTICES = [np.array([[10, 500], [10, 250], [399, 200], [401, 200], [800, 250], [800, 500]])]
                          #bot left  mid left   top left    top right   mid right   bot right
LINE_COLOR = [255, 255, 255] # white
LINE_WIDTH = 3 # width in pixels for drawing lines on image

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
    #processed_image = original_image
    processed_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    #processed_image = _region_of_interest(processed_image, ROI_VERTICES)

    # edge detection
    #processed_image = cv2.Canny(processed_image, threshold1=100, threshold2=200, edges=None, apertureSize=3)

    #lines = cv2.HoughLinesP(image=processed_image, rho=1, theta=np.pi/180,
    #                        threshold=60, lines=None, minLineLength=100, maxLineGap=5)
    #_draw_lines(processed_image, lines)

    processed_image = cv2.pyrDown(src=processed_image, dstsize=(400,300))
    processed_image = cv2.pyrDown(src=processed_image, dstsize=(200,150))
    processed_image = cv2.pyrDown(src=processed_image, dstsize=(100,75))

    return processed_image


def calculate_optical_flow(last_screen, next_screen, last_flow):
    prvs = cv2.cvtColor(last_screen,cv2.COLOR_BGR2GRAY)
    next_ = cv2.cvtColor(next_screen,cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev=prvs, next=next_, flow=last_flow,
                                        pyr_scale=0.5, levels=1, winsize=50,
                                        iterations=3, poly_n=7, poly_sigma=1.5,
                                        flags=0)
    flow_x, flow_y = flow[...,0], flow[...,1]
    
    leftside_mask = np.zeros_like(prvs)
    leftside_mask[:, :int(leftside_mask.shape[1]/2)] = 255
    left_flow_in_left_direction = np.where(flow_x * leftside_mask < 0, flow_x, 0)

    rightside_mask = np.zeros_like(prvs)
    rightside_mask[:, int(leftside_mask.shape[1]/2):] = 255
    right_flow_in_right_direction = np.where(flow_x * rightside_mask > 0, flow_x, 0)

    flow_in_downward_direction = np.where(flow_y < 0, flow_y, 0)
    total_flow = np.abs(flow_in_downward_direction.mean().mean()) + \
                 np.abs(left_flow_in_left_direction.mean().mean()) + \
                 np.abs(right_flow_in_right_direction.mean().mean())
    return total_flow, flow


def get_user_input():
    """
    Returns np.array with boolean values for whether w, a, s, d, space are pressed.
    indices: 0=w, 1=a, 2=s, 3=d, 4=space, 5=c
    """

    bool_list = [keyboard.is_pressed("w"), keyboard.is_pressed("a"),
                 keyboard.is_pressed("s"), keyboard.is_pressed("d"),
                 keyboard.is_pressed("space"), keyboard.is_pressed("c")]
    float_array = np.array([float(1.0) if k else float(0.0) for k in bool_list])
    
    return float_array


def send_input(prediction):
    #print("sending input")
    if prediction[0] > 0.4:
        pydirectinput.keyDown('w', _pause=False)
    else: 
        pydirectinput.keyUp('w', _pause=False)

    if prediction[1] > 0.4:
        pydirectinput.keyDown('a', _pause=False)
    else: 
        pydirectinput.keyUp('a', _pause=False)
        
    if prediction[2] > 0.4:
        pydirectinput.keyDown('s', _pause=False)
    else: 
        pydirectinput.keyUp('s', _pause=False)
        
    if prediction[3] > 0.4:
        pydirectinput.keyDown('d', _pause=False)
    else: 
        pydirectinput.keyUp('d', _pause=False)
    #print("input sent!")


#def send_input_single_key(prediction):
#    hexKeyCode = KEYBOARD_MAPPING[key]
#    extra = ctypes.c_ulong(0)
#    ii_ = Input_I()
#    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra))
#    x = Input( ctypes.c_ulong(1), ii_)
#   SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
#    return


def send_input_single_key(prediction):
    """

    Parameters
    ----------
    prediction : int-like 0=forward, 1=forward and left, 2=left, ... and so on around the d-pad

    Returns
    -------
    Doesn't return anything, just sends input
    """
    keys = ['w', 'a', 's', 'd']
    if prediction == 0:  # forward
        pydirectinput.keyDown('w', _pause=False)
        [pydirectinput.keyUp(key, _pause=False) for key in keys if key != 'w']
    elif prediction == 1:  # forward and left
        pydirectinput.keyDown('w', _pause=False)
        pydirectinput.keyDown('a', _pause=False)
        [pydirectinput.keyUp(key, _pause=False) for key in keys if key not in ['w', 'a']]
    elif prediction == 2:  # left
        pydirectinput.keyDown('a', _pause=False)
        [pydirectinput.keyUp(key, _pause=False) for key in keys if key != 'a']
    elif prediction == 3:  # reverse and left
        pydirectinput.keyDown('s', _pause=False)
        pydirectinput.keyDown('a', _pause=False)
        [pydirectinput.keyUp(key, _pause=False) for key in keys if key not in ['s', 'a']]
    elif prediction == 4:  # reverse
        pydirectinput.keyDown('s', _pause=False)
        [pydirectinput.keyUp(key, _pause=False) for key in keys if key != 's']
    elif prediction == 5:  # reverse and right
        pydirectinput.keyDown('s', _pause=False)
        pydirectinput.keyDown('d', _pause=False)
        [pydirectinput.keyUp(key, _pause=False) for key in keys if key not in ['s', 'd']]
    elif prediction == 6:  # right
        pydirectinput.keyDown('d', _pause=False)
        [pydirectinput.keyUp(key, _pause=False) for key in keys if key != 'd']
    elif prediction == 7:  # forward and left
        pydirectinput.keyDown('w', _pause=False)
        pydirectinput.keyDown('d', _pause=False)
        [pydirectinput.keyUp(key, _pause=False) for key in keys if key not in ['w', 'd']]


def is_outlier(new_value, previous_values, outlier_constant=1.5):
    """
    Check if new_value is outlier wrt previous_values.

    Parameters
    ----------
    new_value : floatlike, new observation to check whether outlier
    previous_values : np.array of floatlikes, previous observations
    outlier_constant : how far out of the IQR new_value must be. Defaults to 1.5

    Returns
    -------
    bool True if outlier, False otherwise
    """
    upper_quartile = np.percentile(previous_values, 75)
    lower_quartile = np.percentile(previous_values, 25)
    iqr = (upper_quartile - lower_quartile) * outlier_constant

    return new_value > upper_quartile + iqr or new_value < lower_quartile - iqr


class Reward:
    def __init__(self):
        self.flow_values_seen = np.array([])
        self.outlier_constant = 1.5
        self.min_samples_to_check_for_outliers = 20
        self.max_optical_flow = 2.8
        self.forward_bonus = 0.15
        self.reverse_penalty = 0.3

    def get_reward(self, optical_flow, prediction):
        reward = 0.0
        if optical_flow > self.max_optical_flow:
            print('clipping optical flow')
        # Check if it's an outlier. If it is, just return the median of the stored values.
        if len(self.flow_values_seen) > self.min_samples_to_check_for_outliers and \
                is_outlier(optical_flow, self.flow_values_seen, self.outlier_constant):
            print('Skipped outlier flow value of', optical_flow)
            reward = min(np.median(self.flow_values_seen), self.max_optical_flow)
        else:
            self.flow_values_seen = np.append(self.flow_values_seen, optical_flow)
            reward = min(optical_flow, self.max_optical_flow)

        if prediction in (0, 1, 7):
            print('bonus for going forward')
            reward = reward + self.forward_bonus
        elif prediction in (3, 4, 5):
            print('penalty for going in reverse')
            reward = reward  - self.reverse_penalty

        return reward
