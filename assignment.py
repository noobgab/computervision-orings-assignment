import cv2 as cv
import numpy as np
import time
import os
import matplotlib.pyplot as plt

def find_peak(hist):
    peaks = [np.where(hist == max(hist))[0][0]]
    p1 = peaks[0] - 50
    p2 = peaks[0] + 50
    new_arr = [hist[i] for i in range(len(hist)) if i < p1 or i > p2]
    peaks.append(new_arr.index(max(new_arr)))
    peaks.sort()
    return peaks

def get_hist(img):
    hist = np.zeros(256)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            hist[img[i, j]] += 1
    return hist

def get_threshold_val(img):
    hist = get_hist(img[1])
    peaks = find_peak(hist)
    t = peaks[0] + int((peaks[1] - peaks[0]) / 2)
    return t

def threshold(img):
    thr = img[1]
    t = get_threshold_val(img)
    for i in range(0, thr.shape[0]):
        for j in range(0, thr.shape[1]):
            if thr[i, j] > t:
                thr[i, j] = 255
            else:
                thr[i, j] = 0
    return thr

def read_images(loc, ext): # Finds all files of the specified extension (ext) within a specified folder (loc), and returns a list of opencv objects of said files
    ret = []
    for file in os.listdir(loc):
        if file.endswith(ext):
            ret.append([os.path.join(loc + '/', file), cv.imread(os.path.join(loc + '/', file), 0)])
    return ret

def dilation(img, struct):
    ret = img
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i, j] == 255:
                #print(i, j)
                pass
    return ret

def erosion(img, struct):
    pass

def closing(img, struct):
    ret = dilation(img, struct)
    return ret

location = './Orings' # The folder containing all the images
file_ext = '.jpg'

images = read_images(location, file_ext) # Read in all the images, store them in a list
results = [True for item in images] # Create a results list which will show whether each one pass (True) or fail (False) -> Pass until a reason to fail is found

morph_struct = [
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
]

for img in images:
    img_thr = threshold(img)
    img_thr = closing(img_thr, morph_struct)
    cv.imshow('Thresholded: ' + img[0], img_thr)
    cv.waitKey(0)
    cv.destroyAllWindows()