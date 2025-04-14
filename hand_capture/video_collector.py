# =============================
# Imports
# =============================

# Built-in modules
import sys
import os

# Third-Party Modules
import cv2
import mediapipe as mp
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config as cfg

# =============================
# Global Settings
# =============================

mp_holistic = mp.solutions.holistic     # type: ignore
mp_drawing = mp.solutions.drawing_utils # type: ignore

train_video_sequence = 30
test_video_sequence = 6

actions = np.array(cfg.dinamic_letters)

# =============================
# Functions
# =============================

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
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
    )

    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
    )


def extract_keypoints(results):
    left_hand = (
        np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks else np.zeros(21 * 3)
    )

    right_hand = (
        np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks else np.zeros(21 * 3)
    )

    return np.concatenate([left_hand, right_hand])


def create_video_folder(letter):
    video_folder_train = os.path.join(cfg.path_data_train, letter)
    video_folder_test = os.path.join(cfg.path_data_test, letter)

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

    for sequence in range(1, train_video_sequence + 1):
        os.makedirs(os.path.join(video_folder_train, str(dirmax_train + sequence)), exist_ok=True)

    for sequence in range(1, test_video_sequence + 1):
        os.makedirs(os.path.join(video_folder_test, str(dirmax_test + sequence)), exist_ok=True)


def collect_video_data(letter):
    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for sequence in range(1, train_video_sequence + 1):
            for frame_num in range(cfg.frame_sequence):
                ret, frame = cap.read()
                image, results = detection_mediapipe(frame, holistic)
                draw_styled_landmarks(image, results)

                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION - TREINO', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Letra {letter} - Vídeo {sequence} (train)', (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(500)
                else:
                    cv2.putText(image, f'Letra {letter} - Vídeo {sequence} (train)', (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)

                keypoints = extract_keypoints(results)
                np.save(os.path.join(cfg.path_data_train, letter, str(sequence), str(frame_num)), keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        for sequence in range(1, test_video_sequence + 1):
            for frame_num in range(cfg.frame_sequence):
                ret, frame = cap.read()
                image, results = detection_mediapipe(frame, holistic)
                draw_styled_landmarks(image, results)

                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION - TESTE', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Letra {letter} - Vídeo {sequence} (test)', (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(500)
                else:
                    cv2.putText(image, f'Letra {letter} - Vídeo {sequence} (test)', (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)

                keypoints = extract_keypoints(results)
                np.save(os.path.join(cfg.path_data_test, letter, str(sequence), str(frame_num)), keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()


# =============================
# Execution
# =============================
if __name__ == "__main__":
    caption_mode = input('Digite 1 para escolher uma letra ou 2 para capturar todas automaticamente: ')

    if caption_mode == '1':
        letter = input(f'Escolha uma letra dentre {cfg.dinamic_letters}: ').upper()
        if letter in cfg.dinamic_letters:
            create_video_folder(letter)
            collect_video_data(letter)
        else:
            print('Letra inválida.')

    elif caption_mode == '2':
        for letter in cfg.dinamic_letters:
            create_video_folder(letter)
            collect_video_data(letter)
    else:
        print('Opção inválida.')
