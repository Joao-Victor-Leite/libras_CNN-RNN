# =============================
# Imports
# =============================

# Built-in Modules
import os
import sys
import time
import datetime

# Third-Party Modules
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

# Local Modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config as cfg


# =============================
# Global Settings
# =============================

actions = np.array(cfg.dinamic_letters)
EPOCHS = 200
FILE_NAME = 'cnn_model_'
LABEL_MAP = {label: num for num, label in enumerate(actions)}


# =============================
# Functions
# =============================

def get_date_str():
    return str('{date:%d_%m_%Y_%H_%M}').format(date=datetime.datetime.now())

def get_time_min(start, end):
    return (end - start)/60

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
            labels.append(LABEL_MAP[action])
    return np.array(sequences), to_categorical(labels).astype(int)

# =============================
# Execution
# =============================

if __name__ == "__main__":
    print("\n\n ----------------------INICIO --------------------------\n")
    print('[INFO] [INICIO]: ' + get_date_str())

    start = time.time()

    # Data Processing
    X_train, y_train = process_data(cfg.path_data_train)
    X_test, y_test = process_data(cfg.path_data_test)

    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # Model Creation - CNN
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=(cfg.frame_sequence, X_train.shape[2])),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(actions.shape[0], activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            validation_data=(X_test, y_test),
            callbacks=[tb_callback, early_stop]
        )

    # Model evaluation with test data
    print("\n[INFO] Avaliando a CNN...")
    score = model.evaluate(X_test, y_test, verbose=1)
    print('[INFO] Accuracy: %.2f%%' % (score[1]*100), '| Loss: %.5f' % (score[0]))

    # Saving Model Weights
    file_date = get_date_str()
    model_path = f'../models/{FILE_NAME}{file_date}.h5'
    model.save(model_path)
    print(f'[INFO] Modelo salvo em: {model_path}')

    end = time.time()
    print("[INFO] CNN Runtime: %.1f min" % (get_time_min(start, end)))

    # Model Summary
    print('[INFO] Summary: ')
    model.summary()

    # Creating directories for graphics and images
    os.makedirs('../models/graphics', exist_ok=True)
    os.makedirs('../models/image', exist_ok=True)

    # Plot of results
    print("[INFO] Generating loss and accuracy graph")
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, EPOCHS), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, EPOCHS), history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, EPOCHS), history.history["categorical_accuracy"], label="train_acc")
    plt.plot(np.arange(0, EPOCHS), history.history["val_categorical_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig('../models/graphics/' + FILE_NAME + file_date + '.png', bbox_inches='tight')

    # Model Structure Plot
    print('[INFO] Generating image of model architecture')
    plot_model(model, to_file='../models/image/' + FILE_NAME + file_date + '.png', show_shapes=True)

    print('\n[INFO] [FIM]: ' + get_date_str())
    print('\n\n')
