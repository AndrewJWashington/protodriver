from protodriver import utils
import numpy as np
from PIL import ImageGrab
import cv2
import time
import pytesseract

SPEED_LOCATION_ON_SCREEN = (600, 470, 705, 510)
speed_image_shape = (SPEED_LOCATION_ON_SCREEN[2] - SPEED_LOCATION_ON_SCREEN[0],
                     SPEED_LOCATION_ON_SCREEN[3] - SPEED_LOCATION_ON_SCREEN[1])
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


def get_speed(_screen):
    _processed_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    _, _processed_screen = cv2.threshold(_processed_screen, 220, 255, cv2.THRESH_BINARY)
    _speed = pytesseract.image_to_string(_processed_screen,
                                         config='digits')
    try:
        #print(_speed, int(_speed))
        _speed = int(_speed)
    except ValueError as e:
        #print(_speed, 'no int equivalent')
        _speed = -1
    return _processed_screen, _speed


if __name__ == "__main__":
    last_time = time.time()
    screen = np.array(ImageGrab.grab(bbox=SPEED_LOCATION_ON_SCREEN))

    while True:
        # grab screen
        screen = np.array(ImageGrab.grab(bbox=SPEED_LOCATION_ON_SCREEN))

        # process image and display resulting image
        processed_screen, speed = get_speed(screen)
        cv2.imshow('window', processed_screen)

        # display framerate
        fps = 1 / (time.time() - last_time)
        last_time = time.time()
        print(f"Speed: {speed:>3} Framerate: {fps:4.4} fps")

        # some stuff to get opencv not to crash
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
