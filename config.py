# =============================
# Configuração Global
# =============================

# Built-in Modules
import string

# =============================
# Caminhos para Dados
# =============================

path_data_train = 'dataset/lstm/train'
path_data_test = 'dataset/lstm/test'

# =============================
# Letras Estáticas e Dinâmicas
# =============================

# Letras com gestos estáticos
static_letters = [letter for letter in string.ascii_uppercase if letter not in ['H', 'J', 'X', 'Y', 'Z']]

# Letras com gestos dinâmicos
dinamic_letters = ['H', 'J', 'X', 'Y', 'Z']

# =============================
# Parâmetros de Coleta de Dados
# =============================

train_video_sequence = 40  # Quantidade de vídeos por letra para treino
test_video_sequence = 10   # Quantidade de vídeos por letra para teste
frame_sequence = 30        # Quantidade de frames por vídeo
