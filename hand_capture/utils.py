import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic     # type: ignore
mp_drawing = mp.solutions.drawing_utils # type: ignore

def draw_landmarks(image, results):
    """
    Desenha os pontos de referência (landmarks) das mãos esquerda e direita na imagem.

    Args:
        image (np.ndarray): Imagem onde os landmarks serão desenhados.
        results (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList): 
            Resultados do modelo MediaPipe contendo os landmarks detectados.
    """

    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def detection_mediapipe(image, model):
    """
    Aplica o modelo MediaPipe na imagem para detectar landmarks.

    Args:
        image (np.ndarray): Imagem no formato BGR.
        model (mediapipe.solutions.holistic.Holistic): Modelo MediaPipe carregado.

    Returns:
        image (np.ndarray): Imagem convertida de volta para BGR após detecção.
        results (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList):
            Resultados da detecção com os landmarks das mãos.
    """
     
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.setflags(write=False)
    results = model.process(image)
    image.setflags(write=True)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    """
    Extrai os pontos chave (x, y, z) das mãos esquerda e direita dos resultados do MediaPipe.

    Args:
        results (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList):
            Resultados da detecção.

    Returns:
        np.ndarray: Vetor concatenado com os keypoints da mão esquerda e direita.
                    Se nenhuma mão for detectada, retorna vetores de zeros.
    """

    left_hand = (
        np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks else np.zeros(21 * 3)
    )

    right_hand = (
        np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks else np.zeros(21 * 3)
    )

    return np.concatenate([left_hand, right_hand])

def draw_styled_landmarks(image, results):
    """
    Desenha os landmarks das mãos esquerda e direita com estilos personalizados.

    Args:
        image (np.ndarray): Imagem onde os landmarks serão desenhados.
        results (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList): 
            Resultados contendo os landmarks detectados.
    """

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