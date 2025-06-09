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

# =============================
# Functions
# =============================

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

    for sequence in range(1, cfg.train_video_sequence + 1):
        os.makedirs(os.path.join(video_folder_train, str(dirmax_train + sequence)), exist_ok=True)

    for sequence in range(1, cfg.test_video_sequence + 1):
        os.makedirs(os.path.join(video_folder_test, str(dirmax_test + sequence)), exist_ok=True)

    return dirmax_train, dirmax_test


def collect_video_data(letter, dirmax_train=0, dirmax_test=0):
    cap = cv2.VideoCapture(0)
    is_dinamic = letter in cfg.dinamic_letters

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for phase in ['train', 'test']:
            total_sequence = cfg.train_video_sequence if phase == 'train' else cfg.test_video_sequence
            dirmax = dirmax_train if phase == 'train' else dirmax_test

            for sequence in range(1, total_sequence + 1):
                current_sequence = dirmax + sequence

                for frame_num in range(cfg.frame_sequence):
                    ret, frame = cap.read()
                    image, results = u.detection_mediapipe(frame, holistic)
                    u.draw_styled_landmarks(image, results)

                    label = 'TREINO' if phase == 'train' else 'TESTE'
                    text_color = (0, 255, 0) if phase == 'train' else (255, 255, 0)

                    if frame_num == 0:
                        cv2.putText(image, f'STARTING COLLECTION - {label}', (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 4, cv2.LINE_AA)

                    cv2.putText(image, f'Letra {letter} - Video {current_sequence} ({phase})', (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                    cv2.imshow('OpenCV Feed', image)

                    if frame_num == 0 and is_dinamic:
                        cv2.waitKey(500)

                    keypoints = u.extract_keypoints(results)  # Agora SEM parâmetro de mão
                    save_path = os.path.join(cfg.path_data_train if phase == 'train' else cfg.path_data_test,
                                             letter, str(current_sequence), str(frame_num))
                    np.save(save_path, keypoints)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

    cap.release()
    cv2.destroyAllWindows()


# =============================
# Execution
# =============================
if __name__ == "__main__":
    caption_mode = input('Digite 1 para escolher uma letra ou 2 para capturar todas automaticamente: ')

    if caption_mode == '1':
        letter = input(f'Escolha uma letra dentre {cfg.static_letters + cfg.dinamic_letters}: ').upper()
        if letter in cfg.static_letters + cfg.dinamic_letters:
            dirmax_train, dirmax_test = create_video_folder(letter)
            collect_video_data(letter, dirmax_train, dirmax_test)
        else:
            print('Letra inválida.')

    elif caption_mode == '2':
        for letter in cfg.static_letters + cfg.dinamic_letters:
            dirmax_train, dirmax_test = create_video_folder(letter)
            collect_video_data(letter, dirmax_train, dirmax_test)
    else:
        print('Opção inválida.')