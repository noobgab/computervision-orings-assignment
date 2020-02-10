import cv2 as cv
import numpy as np
import time
import os
import matplotlib.pyplot as plt

def find_peaks(hist):
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
    peaks = find_peaks(hist)
    t = peaks[0] + int((peaks[1] - peaks[0]) / 2)
    return t

def threshold(img):
    thr = img[1].copy()
    t = get_threshold_val(img)
    for i in range(0, thr.shape[0]):
        for j in range(0, thr.shape[1]):
            if thr[i, j] > t:
                thr[i, j] = 255
            else:
                thr[i, j] = 0
    return thr

def read_images(loc, ext):
    ret = []
    for file in os.listdir(loc):
        if file.endswith(ext):
            ret.append([os.path.join(loc + '/', file), cv.imread(os.path.join(loc + '/', file), 0)])
    return ret

def dilation(img, struct):
    ret = img.copy()
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i, j] == 255:
                for y in range(-1, 2):
                    for x in range(-1, 2):
                        new_i = i + y
                        new_j = j + x
                        if y != 0 and x != 0 and new_i >= 0 and new_i < img.shape[0] and new_j >= 0 and new_j < img.shape[1] and struct[y][x] == 1 and img[new_i, new_j] == 0:
                            ret[i, j] = 0
    return ret

def erosion(img, struct):
    ret = img.copy()
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i, j] == 0:
                for y in range(-1, 2):
                    for x in range(-1, 2):
                        new_i = i + y
                        new_j = j + x
                        if y != 0 and x != 0 and new_i >= 0 and new_i < img.shape[0] and new_j >= 0 and new_j < img.shape[1] and struct[y][x] == 1 and img[new_i, new_j] == 255:
                            ret[i, j] = 255
    return ret

def closing(img, struct):
    ret = dilation(img, struct)
    ret = erosion(ret, struct)
    return ret

def label_components(img):
    labels = [[0 for j in range(0, img.shape[1])] for i in range(0, img.shape[0])]
    cur_lab = 1
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i, j] == 0 and labels[i][j] == 0:
                labels[i][j] = cur_lab
                q = []
                q.append([i, j])

                while len(q) > 0:
                    item = q.pop(0)
                    if img[item[0] - 1, item[1]] == 0 and labels[item[0] - 1][item[1]] == 0:
                        q.append([item[0] - 1, item[1]])
                        labels[item[0] - 1][item[1]] = cur_lab
                    if img[item[0] + 1, item[1]] == 0 and labels[item[0] + 1][item[1]] == 0:
                        q.append([item[0] + 1, item[1]])
                        labels[item[0] + 1][item[1]] = cur_lab
                    if img[item[0], item[1] - 1] == 0 and labels[item[0]][item[1] - 1] == 0:
                        q.append([item[0], item[1] - 1])
                        labels[item[0]][item[1] - 1] = cur_lab
                    if img[item[0], item[1] + 1] == 0 and labels[item[0]][item[1] + 1] == 0:
                        q.append([item[0], item[1] + 1])
                        labels[item[0]][item[1] + 1] = cur_lab
                cur_lab += 1
    return labels

def calc_component_area(labels, label):
    area = 0
    for y in range(0, len(labels)):
        for x in range(0, len(labels[0])):
            if labels[y][x] == label:
                area += 1
    return area

def update_labels(img, labels):
    ret = img.copy()
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            ret[i, j] = labels[i][j] * 70
    return ret

def remove_smallest_areas(labels):
    u = np.unique(labels)
    if len(u) > 2:
        u = u[1:]
        areas = []
        for i in range(len(u)):
            areas.append(calc_component_area(labels, u[i]))
        smallest_area = u[areas.index(min(areas))]
        new_labels = []
        for label_set in labels:
            new_set = []
            for item in label_set:
                if item == smallest_area:
                    new_set.append(0)
                else:
                    new_set.append(item)
            new_labels.append(new_set)
        return new_labels
    return labels

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
    img_labels = label_components(img_thr)
    img_labels = remove_smallest_areas(img_labels)
    img_thr = update_labels(img_thr, img_labels)
    cv.imshow('Thresholded: ' + img[0], img_thr)
    cv.waitKey(0)
    cv.destroyAllWindows()