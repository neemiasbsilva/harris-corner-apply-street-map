
import cv2
import numpy as np
from skimage.io import imread

class HarrisCornerDetector:

    def __init__(self, image_bgr, image_gray, kernel_size):
        
        self.image_bgr = image_bgr
        self.image_gray = image_gray
        self.kernel_size = kernel_size

    def generator(self):
        
        gray = np.float32(self.image_gray)

        dst = cv2.cornerHarris(gray, 2, 3, 0.04)


        #result is dilated for marking the corners, not important
        dst = cv2.dilate(dst, None)

        # Threshold for an optimal value, it may vary depending on the image.
        img = self.image_bgr.copy()

        img[dst > 0.01*dst.max()] = [255, 0, 0]
        
        return img
