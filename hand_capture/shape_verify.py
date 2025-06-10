import numpy as np
import os

base_dir = "dataset/lstm/train/O/1"

for frame_num in range(30):
    file_path = os.path.join(base_dir, f"{frame_num}.npy")
    if os.path.exists(file_path):
        data = np.load(file_path)
        print(f"\nFrame {frame_num:02d} - shape: {data.shape}")
        print("=" * 90)
        print(f"{'Mão Direita':^45}{'|':^5}{'Mão Esquerda':^45}")
        print("-" * 90)
        
        if data.shape == (126,):
            landmarks = data.reshape(42, 3)
            
            # Cabeçalho das colunas
            print(f"{'Landmark':<8}{'X':>10}{'Y':>12}{'Z':>12}   |   {'Landmark':<8}{'X':>10}{'Y':>12}{'Z':>12}")
            
            for i in range(21):
                # Dados da mão direita (primeiros 21 landmarks)
                x_r, y_r, z_r = landmarks[i]
                right_str = f"L{i:02d}: {x_r:>10.6f}{y_r:>12.6f}{z_r:>12.6f}" if np.any([x_r, y_r, z_r]) else f"L{i:02d}: {'Não detectado':>36}"
                
                # Dados da mão esquerda (últimos 21 landmarks)
                x_l, y_l, z_l = landmarks[i+21]
                left_str = f"L{i:02d}: {x_l:>10.6f}{y_l:>12.6f}{z_l:>12.6f}" if np.any([x_l, y_l, z_l]) else f"L{i:02d}: {'Não detectado':>36}"
                
                print(f"{right_str}   |   {left_str}")
        else:
            print("Formato não suportado. Dados crus:")
            print(data)
            
        print("=" * 90)
    else:
        print(f"Arquivo não encontrado: {file_path}")