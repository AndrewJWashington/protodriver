import numpy as np
import cv2
import pyautogui
import pydirectinput
import keyboard
import pytesseract
from collections import deque


# config
                                  # bot left  mid left   top left    top right   mid right   bot right
ROI_VERTICES =          [np.array([[10, 500], [10, 250], [399, 200], [401, 200], [800, 250], [800, 500]])]
ROI_VERTICES_BASELINE = [np.array([[10, 350], [10, 300], [300, 225], [500, 225], [800, 300], [800, 350]])]
LINE_COLOR = [255, 255, 255]  # white
LINE_WIDTH = 3  # width in pixels for drawing lines on image
SPEED_LOCATION_ON_SCREEN = (600, 430, 705, 470)
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


def _region_of_interest_grayscale(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(image, mask)


def _region_of_interest_color(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, (255, 255, 255))
    return cv2.bitwise_and(image, mask)


def alv_vision(image, rgb, thresh):
    '''
    Source: https://github.com/stephencwelch/self_driving_cars/blob/master/notebooks/Self-Driving%20Cars%20%5BPart%201-%20The%20ALV%5D.ipynb
    Apply the basic color-based road segmentation algorithm used in
    the autonomous land vehicle.
    Args
    image: color input image, dimension (n,m,3)
    rgb: tri-color operation values, dimension (3)
    thresh: threshold value for road segmentation

    Returns
    mask: binary mask of the size (n, m), ones indicate road, zeros indicate non-road
    '''
    print(image.shape[0], image.shape[1], image.shape[2])
    dot = np.dot(image.reshape(-1, 3), rgb) > thresh
    return dot.reshape(image.shape[0], image.shape[1])


def process_image_alv(original_image):
    """
    Adapted from https://github.com/stephencwelch/self_driving_cars/blob/master/notebooks/Self-Driving%20Cars%20%5BPart%201-%20The%20ALV%5D.ipynb
    overlays
    """
    processed_image = original_image
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

    # Run alv vision algorithm
    # Tunables: rgb (which colors you're trying to separate) and thresh (where to set the boundary)
    mask = alv_vision(processed_image, rgb=[1, 0, -1], thresh=-50)

    # Display mask on grayscale version of original image
    im_gray = np.tile(np.expand_dims(cv2.cvtColor(processed_image,
                                                  cv2.COLOR_RGB2GRAY)
                                     , axis=2)
                      , (1, 1, 3))

    # Shade road pixels
    im_gray[:, :, 1][mask] = 0.5 * im_gray[:, :, 1][mask] + 255 * 0.5
    im_gray[:, :, 2][mask] = 0.5 * im_gray[:, :, 2][mask] + 255 * 0.5

    return im_gray


plot_scanlines = False
if plot_scanlines:
    import matplotlib.pyplot as plt
    fig = plt.gcf()
    fig.show()

def process_image_for_baseline(original_image):
    processed_image = original_image
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    #processed_image = _region_of_interest_color(processed_image, ROI_VERTICES_BASELINE)

    x1 = 200
    y1 = 260
    x2 = 10
    y2 = 400
    x3 = processed_image.shape[1] - x1
    y3 = y1
    y4 = y2
    x4 = x3 + (x1 - x2)

    x1_final = 200
    x2_final = x1_final
    y1_final = 0
    y2_final = 200
    x3_final = 2 * x1_final
    x4_final = x3_final
    y3_final = y1_final
    y4_final = y2_final
    src_points = np.array([[x1, y1],
                           [x2, y2],
                           [x3, y3],
                           [x4, y4]])
    dst_points = np.array([[x1_final, y1_final],
                           [x2_final, y2_final],
                           [x3_final, y3_final],
                           [x4_final, y4_final]])
    # Uncomment if you want to see the trapezoid
    # cv2.line(processed_image, (x1, y1), (x2, y2), LINE_COLOR, LINE_WIDTH)
    # cv2.line(processed_image, (x2, y2), (x4, y4), LINE_COLOR, LINE_WIDTH)
    # cv2.line(processed_image, (x3, y3), (x4, y4), LINE_COLOR, LINE_WIDTH)
    # cv2.line(processed_image, (x1, y1), (x3, y3), LINE_COLOR, LINE_WIDTH)
    # return processed_image, 10000000, 'left'

    # crop image
    trapezoid = [np.array([[x1, y1], [x2, y2], [x4, y4], [x3, y3]])]
    cropped = _region_of_interest_color(processed_image, trapezoid)

    # warp image and convert to grayscale
    H, _ = cv2.findHomography(src_points, dst_points)
    warped = cv2.warpPerspective(cropped, H, dsize=(3*(x3_final - x1_final), y2_final - y1_final))
    warped = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)

    if plot_scanlines:
        plt.clf()
        scanline, sI_shifted, score = compute_scanline(shifted_image, mask_small, same_padding_columns=5)
        plt.plot(scanline, linewidth=4.0, color='k')
        for top_index in sI_shifted[-5:]:
            plt.plot((top_index, top_index + 1),
                     (scanline[top_index], scanline[top_index + 1]), c=[0, 1, 1], linewidth=4)
        fig.canvas.draw()

    shifted_image, mask_small, scanline, sI_shifted, scores, curvature, direction = \
        find_optimal_unwrapping(warped,
                                steps=3,
                                min_radius=100,
                                max_radius=1e4,
                                k=100,
                                x_padding=x1_final)

    print(curvature, direction)

    return shifted_image, curvature, direction


def unroll_image(warped_small_gray, direction, R, k, x_padding):
    '''
    Unroll warped_small_gray image.
    direction = left or right
    R = radius of circle
    k = distance from top of image to car
    x_padding = amount of zeros padding on the left and right of warped_small_gray
    '''

    W = warped_small_gray.shape[1]  # Image width
    H = warped_small_gray.shape[0]  # Image heigth

    if direction == 'left':
        h = W / 2 - R
    elif direction == 'right':
        h = W / 2 + R
    else:
        print('direction not implemented')

    shifts = np.zeros(warped_small_gray.shape[0])
    for y in range(len(shifts)):
        if direction == 'right':
            shifts[y] = np.sqrt(R ** 2 - (y - k) ** 2) - R
        elif direction == 'left':
            shifts[y] = R - np.sqrt(R ** 2 - (y - k) ** 2)

    shifts = shifts.round().astype('int')

    shifted_image = np.zeros_like(warped_small_gray)
    mask_small = np.zeros_like(warped_small_gray)

    for y in range(shifted_image.shape[0]):
        shifted_image[y, x_padding + shifts[y]:-(x_padding - shifts[y])] = warped_small_gray[y, x_padding:-x_padding]
        mask_small[y, x_padding + shifts[y]:-(x_padding - shifts[y])] = 255

    return shifted_image, mask_small


def compute_scanline(shifted_image, mask_small, same_padding_columns=5):
    '''
    Compute scanline and top 5 largest adjacent differences, and compute score
    same_padding_columns = number of columns to pad on each side of mask.
    '''

    overhead_nan_image = np.copy(shifted_image).astype('float32')
    overhead_nan_image[np.logical_not(mask_small)] = np.NaN

    # Add padding
    for i in range(overhead_nan_image.shape[0]):
        for j in range(overhead_nan_image.shape[1] - 1):
            if np.isnan(overhead_nan_image[i, j]) and not np.isnan(overhead_nan_image[i, j + 1]):
                overhead_nan_image[i, j - same_padding_columns:j + 1] = overhead_nan_image[i, j + 1]
                break;

        for j in range(overhead_nan_image.shape[1] - 1):
            if not np.isnan(overhead_nan_image[i, j]) and np.isnan(overhead_nan_image[i, j + 1]):
                overhead_nan_image[i, j:j + same_padding_columns] = overhead_nan_image[i, j]
                break;

    scanline = np.sum(overhead_nan_image, axis=0)

    scanline_trimmed = scanline[~np.isnan(scanline)]

    d = np.diff(scanline_trimmed.astype('float'))
    sI = np.argsort(abs(d))
    top_5 = d[sI[-5:]]
    score = sum(abs(top_5))

    sI_shifted = np.where(~np.isnan(scanline))[0][sI]

    return scanline, sI_shifted, score


def one_over_mapping(start, end, numpoints):
    start_inverse = 1.0 / start
    end_inverse = 1.0 / end

    inverse_points = np.linspace(start_inverse, end_inverse, numpoints)
    return 1.0 / inverse_points


def find_optimal_unwrapping(warped_small_gray,
                            steps=512,
                            min_radius=150,
                            max_radius=1e4,
                            same_padding_columns=5,
                            k=100,
                            x_padding=64):
    # For the sake of simplicity, assume 1 m = 1 pixel

    # Curvature Hypotheses
    radii = np.concatenate(
        (one_over_mapping(min_radius, max_radius, steps), [1e6], one_over_mapping(max_radius, min_radius, steps)))
    directions = ['right'] * (steps + 1)
    directions.extend(['left'] * steps)

    scores = []

    for i, R in enumerate(radii):  # (tqdm(radii)):
        direction = directions[i]

        shifted_image, mask_small = unroll_image(warped_small_gray,
                                                 direction=direction,
                                                 R=R,
                                                 k=k,
                                                 x_padding=x_padding)

        scanline, sI_shifted, score = compute_scanline(shifted_image, mask_small, same_padding_columns)
        scores.append(score)

    scores = np.array(scores)
    winning_index = np.argmax(scores)

    R = radii[winning_index]
    direction = directions[winning_index]

    shifted_image, mask_small = unroll_image(warped_small_gray,
                                             direction=direction,
                                             R=R,
                                             k=k,
                                             x_padding=x_padding)

    scanline, sI_shifted, score = compute_scanline(shifted_image, mask_small, same_padding_columns)

    return shifted_image, mask_small, scanline, sI_shifted, scores, R, direction


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


def get_speed(screen):
    _processed_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    _processed_screen = _processed_screen[
                        SPEED_LOCATION_ON_SCREEN[1]:SPEED_LOCATION_ON_SCREEN[3],
                        SPEED_LOCATION_ON_SCREEN[0]:SPEED_LOCATION_ON_SCREEN[2]]
    _, _processed_screen = cv2.threshold(_processed_screen, 220, 255, cv2.THRESH_BINARY)
    #  cv2.imshow('window', _processed_screen)
    _speed = pytesseract.image_to_string(_processed_screen, config='digits')

    try:
        _speed = int(_speed)
    except ValueError as e:
        _speed = -1

    return _speed


def _is_valid_speed(speed, speed_values_seen):
    return isinstance(speed, int) and 0 <= speed < 200  # mph


class Reward:
    def __init__(self):
        #  self.flow_values_seen = np.array([])
        self.speed_values_seen = deque(maxlen=3)
        self.outlier_constant = 1.5
        self.min_samples_to_check_for_outliers = 20
        self.max_optical_flow = 2.8
        self.forward_bonus = 4
        self.reverse_penalty = None  # not yet implemented
        self.brake_penalty = 2

    def get_reward(self, optical_flow, speed, prediction):
        reward = 0.0
        if _is_valid_speed(speed, self.speed_values_seen):
            reward = float(speed)
            self.speed_values_seen.append(speed)
        else:
            reward = float(np.mean(self.speed_values_seen))

        if prediction in (0, 1, 7) and reward > 5:
            reward = reward + self.forward_bonus
        elif prediction in (3, 4, 5):
            reward = reward - self.brake_penalty
        #  todo - penalize more if in reverse. Can't just check for key because brake is same as reverse
        #  instead we can look for an "R" with tesseract

        return reward
