import cv2
import numpy as np



def show_angle(image, angle, point, width, height,identifier):
    return cv2.putText(image, str(int(angle)), 
                           tuple(np.multiply(point, [width, height]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (250,250,250), 1, cv2.LINE_AA
                                )