import cv2
import time
import numpy as np
import os
import string

image_x, image_y = 64, 64
dataset_train_size = 800
dataset_test_size = 200
dataset_size = dataset_train_size + dataset_test_size

cap = cv2.VideoCapture(0)

dir_img_train = './pre_processed/train/'
dir_img_test = './pre_processed/test/'

if not os.path.exists(dir_img_train) and not os.path.exists(dir_img_test):
    os.makedirs(dir_img_train)
    os.makedirs(dir_img_test)

static_letters = [letter for letter in string.ascii_uppercase if letter in ['H', 'J', 'X', 'Y', 'Z']]
