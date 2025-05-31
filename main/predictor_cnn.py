import os
import cv2
import sys
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model


current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.abspath(os.path.join(current_dir, '..', 'models'))


def carregar_modelo(model_dir):

    if not os.path.exists(model_dir):
        print(f"❌ Diretório não encontrado: {model_dir}")
        sys.exit()

    h5_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]

    if not h5_files:
        print("❌ Nenhum arquivo .h5 encontrado no diretório especificado.")
        sys.exit()

    print("\n📦 Modelos disponíveis:")
    for idx, fname in enumerate(h5_files, 1):
        print(f"{idx}: {fname}")

    try:
        index = int(input("\nDigite o número do modelo que deseja carregar: ")) - 1

        if index < 0 or index >= len(h5_files):
            print("❌ Índice inválido.")
            sys.exit()

        model_path = os.path.join(model_dir, h5_files[index])
        print(f"\n🚀 Carregando modelo: {model_path}")

        model = load_model(model_path)

        labels_dict = {
            0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G',
            7: 'I', 8: 'J', 9: 'K', 10: 'L', 11: 'M', 12: 'N',
            13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T',
            19: 'U', 20: 'V', 21: 'W'
        }

        return model, labels_dict, model_path

    except ValueError:
        print("❌ Entrada inválida. Digite um número válido.")
        sys.exit()


model, labels_dict, model_path = carregar_modelo(model_dir)


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

print("🎥 Iniciando câmera. Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Falha ao capturar imagem da câmera.")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]

            if x_ and y_:  # Garante que não está vazio
                # Coordenadas normalizadas
                base_x = hand_landmarks.landmark[0].x
                base_y = hand_landmarks.landmark[0].y
                base_z = hand_landmarks.landmark[0].z

                data_aux = []
                for lm in hand_landmarks.landmark:
                    x = lm.x - base_x
                    y = lm.y - base_y
                    z = lm.z - base_z
                    data_aux.append([lm.x, lm.y, lm.z, x, y, z])

                data_aux = np.array(data_aux).reshape(1, 21, 6)

                prediction = model.predict(data_aux, verbose=0)
                predicted_index = np.argmax(prediction)
                predicted_char = labels_dict.get(predicted_index, '?')

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) + 10
                y2 = int(max(y_) * H) + 10

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, predicted_char, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)

            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

    cv2.imshow('Predicao em tempo real', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
