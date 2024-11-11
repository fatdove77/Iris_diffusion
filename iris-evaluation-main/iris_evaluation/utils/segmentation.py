"""
Written by Rob Hart of Walsh Lab @ IU Indianapolis.
"""

import numpy as np
import cv2

def otsu_thresh(image):
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        image_bright_raw = np.power(image_gray, 0.5)
        image_bright = cv2.normalize(image_bright_raw, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        image_blur = cv2.GaussianBlur(image_bright, (5, 5), 0) 
        image_thresh = cv2.threshold(image_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        return image_thresh #mask

def elementary_segmentation(img_thresh):
    shape_img = img_thresh.shape[0]//2
    contours, _ = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
    
    limbic_range = (shape_img-3, shape_img+3)
    pupilary_range = ((shape_img*0.33)-3, (shape_img*0.33)+3)
    center_range = ((shape_img - 3), (shape_img + 3))
    
    circles = []

    for i, contour in enumerate(contours):
        circle = cv2.minEnclosingCircle(contour)
        circles.append(np.array([circle[0][0], circle[0][1], circle[1]], dtype='float32'))

    circles = np.array(circles)

    conditions_limbic = (circles[:, 2] >= limbic_range[0]) & (circles[:, 2] <= limbic_range[1]) & (circles[:, 0] >= center_range[0]) & (circles[:, 0] <= center_range[1]) & (circles[:, 1] >= center_range[0]) & (circles[:, 1] <= center_range[1])
    conditions_pupillary = (circles[:, 2] >= pupilary_range[0]) & (circles[:, 2] <= pupilary_range[1]) & (circles[:, 0] >= center_range[0]) & (circles[:, 0] <= center_range[1]) & (circles[:, 1] >= center_range[0]) & (circles[:, 1] <= center_range[1])
    
    limbic_circle = circles[conditions_limbic]
    pupillary_circle = circles[conditions_pupillary]


    return limbic_circle, pupillary_circle