from protodriver import utils
import numpy as np
from PIL import ImageGrab
import cv2

# heavily borrowed from https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html

screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))
last_processed_screen = screen #utils.process_image(screen)

prvs = cv2.cvtColor(last_processed_screen,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(last_processed_screen)
last_flow = np.zeros_like(last_processed_screen)
hsv[...,1] = 255

def demo_optical_flow(last_screen, next_screen, last_flow):
    prvs = cv2.cvtColor(last_screen,cv2.COLOR_BGR2GRAY)
    next_ = cv2.cvtColor(next_screen,cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev=prvs, next=next_, flow=last_flow,
                                        pyr_scale=0.5, levels=1, winsize=50,
                                        iterations=3, poly_n=7, poly_sigma=1.5,
                                        flags=0)
    flow_x, flow_y = flow[...,0], flow[...,1]
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    print('mag', mag.mean().mean())

    
    leftside_mask = np.zeros_like(prvs)
    leftside_mask[:, :int(leftside_mask.shape[1]/2)] = 255
    #cv2.imshow('window1', leftside_mask) uncomment to show that mask is correct
    left_flow_in_left_direction = np.where(flow_x * leftside_mask < 0, flow_x, 0)

    rightside_mask = np.zeros_like(prvs)
    rightside_mask[:, int(leftside_mask.shape[1]/2):] = 255
    right_flow_in_right_direction = np.where(flow_x * rightside_mask > 0, flow_x, 0)

    flow_in_downward_direction = np.where(flow_y < 0, flow_y, 0)
    total_flow = np.abs(flow_in_downward_direction.mean().mean()) + \
                 np.abs(left_flow_in_left_direction.mean().mean()) + \
                 np.abs(right_flow_in_right_direction.mean().mean())
    print('down flow', flow_in_downward_direction.mean().mean())
    print('left flow', left_flow_in_left_direction.mean().mean())
    print('right flow', right_flow_in_right_direction.mean().mean())

    print('total flow', total_flow)

    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB), flow

if __name__ == "__main__":
    for frames_processed in range(200):
        # grab screen
        screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))

        # process image and display resulting image
        processed_screen = screen #utils.process_image(screen)

        image_to_show, last_flow = demo_optical_flow(last_processed_screen, processed_screen, last_flow)

        cv2.imshow('window', image_to_show)

        last_processed_screen = processed_screen


        # some stuff to get opencv not to crash
        if(cv2.waitKey(25) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break

