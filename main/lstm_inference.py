import os
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'I', 4: 'O',
}
"""
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G',
    7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N',
    14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}
"""

sequence_length = 30

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))
model = None

if not os.path.exists(MODEL_DIR):
    print(f"Diretório não encontrado: {MODEL_DIR}")
    exit(1)
else:
    h5_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.h5') and f.startswith('lstm_model')]
    if not h5_files:
        print("Nenhum arquivo .h5 começando com 'lstm_model' encontrado no diretório especificado.")
        exit(1)
    print("Modelos disponíveis:")
    for idx, fname in enumerate(h5_files, 1):
        print(f"{idx}: {fname}")

    while True:
        try:
            index = int(input("\nDigite o índice do modelo que deseja carregar: ")) - 1
            if index < 0 or index >= len(h5_files):
                print("Índice inválido.")
                continue
            model_path = os.path.join(MODEL_DIR, h5_files[index])
            print(f"\nCarregando modelo: {model_path}")
            model = load_model(model_path)

            if model is None:
                print("❌ Erro: Modelo não foi carregado!")
                exit(1)
            else:
                print("✅ Modelo carregado com sucesso!")
                print("Esperado pelo modelo:", model.input_shape)
            break
        except ValueError:
            print("Por favor, digite um número válido.")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

cap = cv2.VideoCapture(0)
sequence = []

while True:
    data_aux = []
    x_ = []
    y_ = []
    z_ = []

    ret, frame = cap.read()
    if not ret:
        print("Não conseguiu ler frame da câmera!")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        all_hands = results.multi_hand_landmarks
        for hand_idx in range(2):
            if hand_idx < len(all_hands):
                hand_landmarks = all_hands[hand_idx]
                x_, y_, z_ = [], [], []
                for i in range(21):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    z = hand_landmarks.landmark[i].z
                    x_.append(x)
                    y_.append(y)
                    z_.append(z)
                for i in range(21):
                    data_aux.append(x_[i] - min(x_))
                    data_aux.append(y_[i] - min(y_))
                    data_aux.append(z_[i] - min(z_))
            else:
                data_aux.extend([0.0] * (21 * 3))

        for hand_landmarks in all_hands:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        sequence.append(data_aux)
        sequence = sequence[-sequence_length:]

        print("data_aux length:", len(data_aux))
        print("sequence length:", len(sequence))

        if len(sequence) == sequence_length:
            try:
                input_lstm = np.array(sequence)[np.newaxis, ...]
                print("Enviado na inferência:", input_lstm.shape)
                res = model.predict(input_lstm, verbose=0)[0]

                pred_idx = int(np.argmax(res))
                predicted_character = labels_dict[pred_idx]
                prob = res[pred_idx]

                if len(all_hands) > 0:
                    hand_landmarks = all_hands[0]
                    x_vals = [lm.x for lm in hand_landmarks.landmark]
                    y_vals = [lm.y for lm in hand_landmarks.landmark]
                    x1 = int(min(x_vals) * W) - 10
                    y1 = int(min(y_vals) * H) - 10
                    x2 = int(max(x_vals) * W) + 10
                    y2 = int(max(y_vals) * H) + 10

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(
                        frame, f"{predicted_character} ({prob:.2f})",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA
                    )
            except Exception as e:
                print("ERRO DURANTE PREDIÇÃO:", e)
                print("Esperado pelo modelo:", model.input_shape)
                break

    else:
        sequence = []

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()