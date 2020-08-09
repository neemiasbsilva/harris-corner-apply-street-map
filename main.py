import os
import argparse
import cv2
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from scipy import signal as sig
from skimage.color import rgb2gray
from harris_corner_detector import HarrisCornerDetector

class ImageGenerator:

    def __init__(self, path_dataset):
        self.path_dataset = path_dataset

        # self.image_paths = []

    
    
    def generator(self):
        
        image_names = os.listdir(path_dataset)
        images_arr = []
        image_bgr_arr = []
        for image in image_names:
            image_bgra = imread(os.path.join(path_dataset, image))
            image_bgr = cv2.cvtColor(image_bgra, cv2.COLOR_BGRA2BGR)
            image_bgr_arr.append(image_bgr)
            image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            images_arr.append(image_gray)
        
        images_arr = np.array(images_arr)
        image_bgr_arr = np.array(image_bgr_arr)

        return image_bgr_arr, images_arr, image_names



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Application of Haris Corner Detector to street map dataset")
    
    parser.add_argument("-path_dataset", action="store", required=True, help="The dataset of image to find the interest point of each image", dest="path_dataset")
    parser.add_argument("-path_save_dataset", action="store", required=True,
                        help="The path to save the corner detector of each image", dest="path_save_dataset")
    
    arguments = parser.parse_args()

    path_dataset = arguments.path_dataset
    path_save_dataset = arguments.path_save_dataset

    dataset_generator = ImageGenerator(path_dataset)

    image_bgr_arr, dataset, image_names = dataset_generator.generator()

    for id, img in enumerate(dataset):
        hcd = HarrisCornerDetector(image_bgr_arr[id], img, 5)
        img = hcd.generator()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # print(np.expand_dims(threshold_image, axis=2).shape)
        # break
        cv2.imwrite(os.path.join(path_save_dataset, image_names[id]), img)
    
