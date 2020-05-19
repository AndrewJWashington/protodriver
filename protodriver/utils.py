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


def find_lanes():
    pass


def get_user_input():
    """
    Returns np.array with boolean values for whether w, a, s, d, space are pressed.
    indices: 0=w, 1=a, 2=s, 3=d, 4=space
    """

    bool_list = [keyboard.is_pressed("w"), keyboard.is_pressed("a"),
                 keyboard.is_pressed("s"), keyboard.is_pressed("d"),
                 keyboard.is_pressed("space"), keyboard.is_pressed("c")]
    float_array = np.array([float(1.0) if k else float(0.0) for k in bool_list])
    
    return float_array


def send_input(prediction):
    #print("sending input")
    if prediction[0] > 0.4:
        pydirectinput.keyDown('w')
    else: 
        pydirectinput.keyUp('w')

    if prediction[1] > 0.4:
        pydirectinput.keyDown('a')
    else: 
        pydirectinput.keyUp('a')
        
    if prediction[2] > 0.4:
        pydirectinput.keyDown('s')
    else: 
        pydirectinput.keyUp('s')
        
    if prediction[3] > 0.4:
        pydirectinput.keyDown('d')
    else: 
        pydirectinput.keyUp('d')
    #print("input sent!")
