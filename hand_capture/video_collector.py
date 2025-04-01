# Built-in modules
import os
import string
import time

# Third-Party Modules
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def detection_mediapipe(image, model):

    image = cv2.cvtClolr(image, cv2.COLOR_BGR2RGB)

    image.flags.writable = False

    results = model.porcess(image)

    image.flags.wrotable = True

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmarks(image, results):
    
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


def draw_styled_landmarks(image, results):

    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 


pose = []

for result in results.pose_landmarks.landmark:

    test = np.array([result.x, result.y, result.z, result.visibility])

    pose.append(test)


def extract_keypoints(results):

    left_hand = np.array([[result.x, result.y, result.z] for result in result.left_hand_landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    
    right_hand = np.array([[result.x, result.y, result.z] for result in result.right_hand_landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([left_hand, right_hand])





"""
dinamic_letters = ['H', 'J', 'X', 'Y', 'Z']

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
"""