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

train_image_count = 800
test_image_count = 200
total_image_count = train_image_count + test_image_count

# =============================
# Functions
# =============================

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


def get_next_index(directory):
    existing = [f for f in os.listdir(directory) if f.endswith('.npy')]
    if not existing:
        return 0
    return max(int(f.split('.')[0]) for f in existing) + 1


def prepare_directories(letter):
    train_dir = os.path.join(cfg.path_data_train, letter)
    test_dir = os.path.join(cfg.path_data_test, letter)
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
            image, results = detection_mediapipe(frame, holistic)
            draw_styled_landmarks(image, results)

            keypoints = extract_keypoints(results)

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

def detection_mediapipe(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.setflags(write=False)
    results = model.process(image)
    image.setflags(write=True)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


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
