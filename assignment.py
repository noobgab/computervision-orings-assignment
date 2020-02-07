import cv2 as cv
import numpy as np
import time
import os

def threshold(img, t):
    pass

def segment(img):
    pass

def read_images(loc, ext): # Finds all files of the specified extension (ext) within a specified folder (loc), and returns a list of opencv objects of said files
    ret = []
    for file in os.listdir(loc):
        if file.endswith(ext):
            ret.append(cv.imread(os.path.join(loc + '/', file), 0))
    return ret

location = './Orings' # The folder containing all the images
images = read_images(location, '.jpg') # Read in all the images, store them in a list
results = [True for item in images] # Create a results list which will show whether each one pass (True) or fail (False) -> Pass until a reason to fail is found

for img in images:
    cv.imshow('Original Image', img)
    cv.waitKey(0)

print("Done processing, press any key to exit...")
cv.waitKey(0)
cv.destroyAllWindows()

print(results)