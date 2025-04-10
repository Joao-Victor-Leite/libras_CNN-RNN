import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


DATA_PATH_TRAIN = './pre_processed/train/'
DATA_PATH_TEST = './pre_processed/test/'

actions = np.array(['H', 'J', 'X', 'Y', 'Z'])
# sequence_length = 30  

label_map = {label: num for num, label in enumerate(actions)}

def load_sequences(data_path):
    sequences, labels = [], []
    for action in actions:
        for sequence in np.array(os.listdir(os.path.join(data_path, action))).astype(int):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(data_path, action, str(sequence), f"{frame_num}.npy"))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
    return np.array(sequences), to_categorical(labels).astype(int)


X_train, y_train = load_sequences(DATA_PATH_TRAIN)
X_test, y_test = load_sequences(DATA_PATH_TEST)