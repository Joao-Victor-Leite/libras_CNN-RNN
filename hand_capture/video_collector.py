# Built-in modules
import os
import string
from sys import path
import time

# Third-Party Modules
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

path_video_train = './pre_processed/train/'
path_video_test = './pre_processed/test/'

dinamic_letters = ['H', 'J', 'X', 'Y', 'Z']


def detection_mediapipe(image, model):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image.setflags(write=False)

    results = model.process(image)

    image.setflags(write=True)

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

""""
pose = []

for result in results.pose_landmarks.landmark:

    test = np.array([result.x, result.y, result.z, result.visibility])

    pose.append(test)
"""


def extract_keypoints(results):

    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)

    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([left_hand, right_hand])

"""
result_test = extract_keypoints(results)
"""


train_video_sequence = 30
test_video_sequence = 6
frame_sequence = 30
start_folder = 30
no_sequence = 30

actions = np.array(dinamic_letters)

def create_video_folder(letter):

    video_folder_train = os.path.join(path_video_train, letter)
    video_folder_test = os.path.join(path_video_test, letter)

    os.makedirs(video_folder_train, exist_ok=True)
    os.makedirs(video_folder_test, exist_ok=True)

    try:
        dirmax_train = np.max(np.array(os.listdir(video_folder_train)).astype(int))
    except ValueError:
        dirmax_train = 0

    try:
        dirmax_test = np.max(np.array(os.listdir(video_folder_test)).astype(int))
    except ValueError:
        dirmax_test = 0

    for sequence in range(1, no_sequence + 1):
        try:
            os.makedirs(os.path.join(video_folder_train, str(dirmax_train + sequence)))
            os.makedirs(os.path.join(video_folder_test, str(dirmax_test + sequence)))
        except FileExistsError:
            pass


def collect_video_data():

    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:

        for sequence in range(1, no_sequence + 1):
            for frame_num in range(frame_sequence):

                # Lê o quadro da câmera
                ret, frame = cap.read()

                # Detecção com MediaPipe
                image, results = detection_mediapipe(frame, holistic)

                # Desenha os pontos de referência (landmarks)
                draw_styled_landmarks(image, results)

                # Lógica para o primeiro frame
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Collecting frames for {letter} Video Number {sequence}', (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(500)
                else:
                    cv2.putText(image, f'Collecting frames for {letter} Video Number {sequence}', (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)

                # Extrai os keypoints
                keypoints = extract_keypoints(results)

                # Salva os keypoints nos diretórios de treino e teste
                npy_path_train = os.path.join(path_video_train, letter, str(sequence), str(frame_num))
                npy_path_test = os.path.join(path_video_test, letter, str(sequence), str(frame_num))

                np.save(npy_path_train, keypoints)
                np.save(npy_path_test, keypoints)

                # Permite sair com 'q'
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break


caption_mode = input('Digite 1 para escolher uma letra ou 2 para capturar todas as letras automaticamente: ')

if caption_mode == '1':

    letter = input(f'Escolha uma letra dentre {dinamic_letters}: ').upper()

    if letter in dinamic_letters:
        create_video_folder(letter)
        collect_video_data()
    else:
        print('Letra inválida.')

elif caption_mode == '2':
    for letter in dinamic_letters:
        create_video_folder(letter)
else:
    print('Opção inválida.')
