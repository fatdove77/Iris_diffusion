"""
Written by Rob Hart of Walsh Lab @ IU Indianapolis.
"""

import numpy as np
import cv2

def convert_LAB(img, thresh):
    LAB_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(LAB_image)
    mask = thresh > 0
    LAB_values = {
        'L' : np.mean(L[mask]),
        'A' : np.mean(A[mask]),
        'B' : np.mean(B[mask])
    }

    return LAB_values