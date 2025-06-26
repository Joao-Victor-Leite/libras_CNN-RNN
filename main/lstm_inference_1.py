import sys
import os
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'hand_capture')))
from utils import extract_keypoints, draw_styled_landmarks, detection_mediapipe

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

sequence_length = 30
mp_holistic = mp.solutions.holistic
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))
model = None

colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

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
            
            
sequence = []
sentence = []
predictions = []
threshold = 0.5

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        ret, frame = cap.read()

        if not ret or frame is None:
            print("Não conseguiu ler frame da câmera!")
            break

        image, results = detection_mediapipe(frame, holistic)

        draw_styled_landmarks(image, results)
        
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(labels_dict[np.argmax(res)])
            predictions.append(np.argmax(res))
            
            
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                    
                    if len(sentence) > 0: 
                        if labels_dict[np.argmax(res)] != sentence[-1]:
                            sentence.append(labels_dict[np.argmax(res)])
                    else:
                        sentence.append(labels_dict[np.argmax(res)])

            if len(sentence) > 5: 
                sentence = sentence[-5:]

            image = prob_viz(res, labels_dict, image, colors)
            
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('OpenCV Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()