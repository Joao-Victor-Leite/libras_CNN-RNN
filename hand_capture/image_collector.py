# =============================
# Imports
# =============================

# Built-in Modules
import os
import sys

# Third-Party Modules
import cv2
import numpy as np

# Local Modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config as cfg
import hand_capture.utils as u

# =============================
# Global Settings
# =============================

mp_holistic = u.mp_holistic
mp_drawing = u.mp_drawing

train_image_count = 400
test_image_count = 100
total_image_count = train_image_count + test_image_count

# =============================
# Functions
# =============================

def get_next_index(directory):
    existing = [f for f in os.listdir(directory) if f.endswith('.npy')]
    if not existing:
        return 0
    return max(int(f.split('.')[0]) for f in existing) + 1


def prepare_directories(letter):
    train_dir = os.path.join(cfg.cnn_path_data_train, letter)
    test_dir = os.path.join(cfg.cnn_path_data_test, letter)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    return train_dir, test_dir

def capture_static_letter(letter):
    cap = cv2.VideoCapture(0)
    train_dir, test_dir = prepare_directories(letter)

    print(f'Capturando letra {letter}. Pressione "s" para iniciar...')

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    train_index = get_next_index(train_dir)
    test_index = get_next_index(test_dir)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        count = 0
        while count < total_image_count:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            image, results = u.detection_mediapipe(frame, holistic)
            u.draw_styled_landmarks(image, results)

            keypoints = u.extract_keypoints(results)

            if count < train_image_count:
                np.save(os.path.join(train_dir, f'{train_index}.npy'), keypoints)
                train_index += 1
            else:
                np.save(os.path.join(test_dir, f'{test_index}.npy'), keypoints)
                test_index += 1

            cv2.imshow('Frame', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

            count += 1

    print(f'Captura da letra {letter} finalizada. Pressione "n" para continuar.')
    while True:
        if cv2.waitKey(1) & 0xFF == ord('n'):
            break

    cap.release()
    cv2.destroyAllWindows()

# =============================
# Execution
# =============================

if __name__ == "__main__":
    mode = input('Digite 1 para escolher uma letra ou 2 para capturar todas automaticamente: ')
    if mode == '1':
        letter = input(f'Escolha uma letra dentre {cfg.static_letters}: ').upper()
        if letter in cfg.static_letters:
            capture_static_letter(letter)
        else:
            print('Letra inválida.')
    elif mode == '2':
        for letter in cfg.static_letters:
            capture_static_letter(letter)
    else:
        print('Opção inválida.')
