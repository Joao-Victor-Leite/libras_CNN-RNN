# =============================
# Imports
# =============================

# Built-in Modules
import os

# Third-Party Modules
import numpy as np

# =============================
# Configurações
# =============================

base_dir = "dataset/lstm/train/O/1"     # Caminho da sequência a ser exibida, atualize conforme o caminho do teu projeto
frame_total = 30                        # Quantidade de frames por sequência
expected_shape = (126,)                 # Esperado: 2 mãos com 21 landmarks e 3 coordenadas cada (42x3)

# =============================
# Execução
# =============================

"""
- Percorre todos os frames de 0 a `frame_total - 1`.
   - Para cada frame:
     - Tenta carregar o arquivo `.npy` correspondente.
     - Se o arquivo estiver presente e no formato esperado:
       - Os dados são reorganizados em (42, 3), sendo os 21 primeiros pontos da mão direita e os 21 seguintes da mão esquerda.
       - Os landmarks de ambas as mãos são exibidos lado a lado, indicando coordenadas X, Y, Z ou “Não detectado” se todos os valores forem zero.
     - Se o arquivo estiver ausente ou tiver formato diferente do esperado, é exibida uma mensagem de erro.
"""

for frame_num in range(frame_total):
    file_path = os.path.join(base_dir, f"{frame_num}.npy")
    
    if not os.path.exists(file_path):
        print(f"Arquivo não encontrado: {file_path}")
        continue

    data = np.load(file_path)
    print(f"\nFrame {frame_num:02d} - shape: {data.shape}")
    print("=" * 90)
    print(f"{'Mão Direita':^45}{'|':^5}{'Mão Esquerda':^45}")
    print("-" * 90)

    if data.shape == expected_shape:
        landmarks = data.reshape(42, 3)

        print(f"{'Landmark':<8}{'X':>10}{'Y':>12}{'Z':>12}   |   {'Landmark':<8}{'X':>10}{'Y':>12}{'Z':>12}")

        for i in range(21):
            # Mão direita
            x_r, y_r, z_r = landmarks[i]
            right_str = (
                f"L{i:02d}: {x_r:>10.6f}{y_r:>12.6f}{z_r:>12.6f}"
                if np.any([x_r, y_r, z_r])
                else f"L{i:02d}: {'Não detectado':>36}"
            )

            # Mão esquerda
            x_l, y_l, z_l = landmarks[i + 21]
            left_str = (
                f"L{i:02d}: {x_l:>10.6f}{y_l:>12.6f}{z_l:>12.6f}"
                if np.any([x_l, y_l, z_l])
                else f"L{i:02d}: {'Não detectado':>36}"
            )

            print(f"{right_str}   |   {left_str}")
    else:
        print("Formato não suportado. Dados crus:")
        print(data)

    print("=" * 90)
