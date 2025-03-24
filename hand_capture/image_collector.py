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

static_letters = [letter for letter in string.ascii_uppercase if letter not in ['H', 'J', 'X', 'Y', 'Z']]

for letter in static_letters:

    class_dir_train = os.path.join(dir_img_train, letter)
    class_dir_test = os.path.join(dir_img_test, letter)

    if not os.path.exists(class_dir_train) and not os.path.exists(class_dir_test):
        os.makedirs(class_dir_train)
        os.makedirs(class_dir_test)

    print(f'Coletando dados para a letra {letter}')
    print('Pressione "s" para iniciar a captura.')

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        # Mostra apenas a ROI
        roi = frame[100:300, 425:625]
        roi_resized = cv2.resize(roi, (image_x, image_y))
        cv2.imshow('ROI', roi_resized)

        if cv2.waitKey(5) == ord('s'):
            break

    count = 0

    while count < dataset_size:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        # Recorta a ROI e redimensiona
        roi = frame[100:300, 425:625]
        roi_resized = cv2.resize(roi, (image_x, image_y))

        # Mostra apenas a ROI
        cv2.imshow('ROI', roi_resized)
        cv2.waitKey(25)

        if count < dataset_train_size:
            cv2.imwrite(os.path.join(class_dir_train, f'{count}.jpg'), roi_resized)
        else:
            cv2.imwrite(os.path.join(class_dir_test, f'{count - dataset_train_size}.jpg'), roi_resized)

        count += 1

    print(f'Captura para {letter} concluída. Pressione "n" para avançar para a próxima letra.')

    while True:
        key = cv2.waitKey(1)
        if key == ord('n'):
            break

cap.release()
cv2.destroyAllWindows()
