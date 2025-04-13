import config as cfg
import numpy as np
import os

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

actions = np.array(cfg.dinamic_letters)

label_map = {label: num for num , label in enumerate(actions)}

def process_data(data_path):
    sequences, labels = [], []

    for action in actions:
        action_path = os.path.join(data_path, action)

        for sequence in np.array(os.listdir(action_path)).astype(int):
            window = []

            for frame_num in range(cfg.frame_sequence):
                res = np.load(os.path.join(action_path, str(sequence), f"{frame_num}.npy"))

                window.append(res)

            sequences.append(window)
            labels.append(label_map[action])
    return np.array(sequences), to_categorical(labels).astype(int)


X_train, y_train = process_data(cfg.path_data_train)
X_test, y_test = process_data(cfg.path_data_test)

log_dir = os.path.join('Logs')
