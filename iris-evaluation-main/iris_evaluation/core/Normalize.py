"""
Written by Rob Hart of Walsh Lab @ IU Indianapolis.
"""

import numpy as np
import cv2

class CoordinateTransformer(object):
    def __init__(self, img, radius_pupillary, radius_limbic, center, output_width, output_length):
        self.__img = img
        self.__output_width = output_width
        self.__output_length = output_length
        self.__center = center
        self.__numline_length = np.linspace(0, 2*np.pi, output_length) #create evenly spaced coordinates between 0 and 2pi, wherein output_length is the number of coordinates
        self.__numline_width = np.linspace(radius_pupillary, radius_limbic, output_width) #create evenly spaced coordinates between the the radius of the pupillary and lumbic radii, where output_width is the number of coordinates
        self.__transformed = None
        self.set_transformed()

    def set_transformed(self): #only works for square images with even dimensions
        transformed = np.zeros((self.__output_width,  self.__output_length, 3)) #create a blank array for output image where width is the radius and length is theta in a polar cooidinate system
        center = self.__center
        rows = self.__img.shape[0] #gets the number of rows and columns
        cols = self.__img.shape[1]

        for c in range(3): # iterate over each color channel
            for i in range(len(self.__numline_width)): #interate through polar coordinates to find corresponding x and y cartesian coordinates, with it in mind that there are 4 central pixels
                for j in range(len(self.__numline_length)):
                    xi = center + self.__numline_width[i] * np.cos(self.__numline_length[j])
                    yi = center + self.__numline_width[i] * np.sin(self.__numline_length[j])
                    """
                    if self.__numline_length[j] >= 0 and self.__numline_length[j] <= np.pi/2: #this if else statement accounts for the fact that there are 4 central pixels in an even image
                        xi = center + self.__numline_width[i] * np.cos(self.__numline_length[j])
                        yi = center + self.__numline_width[i] * np.sin(self.__numline_length[j])
                    elif self.__numline_length[j] > np.pi/2 and self.__numline_length[j] <= np.pi:
                        xi = (center - 1) + self.__numline_width[i] * np.cos(self.__numline_length[j])
                        yi = center + self.__numline_width[i] * np.sin(self.__numline_length[j])
                    elif self.__numline_length[j] > np.pi and self.__numline_length[j] <= 3*np.pi/2:
                        xi = (center - 1) + self.__numline_width[i] * np.cos(self.__numline_length[j])
                        yi = (center - 1) + self.__numline_width[i] * np.sin(self.__numline_length[j])
                    elif self.__numline_length[j] > 3*np.pi/2 and self.__numline_length[j] <= 2*np.pi:
                        xi = center + self.__numline_width[i] * np.cos(self.__numline_length[j])
                        yi = (center - 1) + self.__numline_width[i] * np.sin(self.__numline_length[j])
                    else:
                        print("ERROR: theta out of bounds")
                    """
                    if 0 <= xi < cols and 0 <= yi < rows: #if the cartesian coordinates exist in the original image is not negative and within the size of the image
                        transformed[i, j, c] = self.__img[int(yi), int(xi), c]
        self.__transformed = transformed.astype(np.uint8)

    def get_transformed(self):
        return self.__transformed

    transformed = property(fget = get_transformed) 
