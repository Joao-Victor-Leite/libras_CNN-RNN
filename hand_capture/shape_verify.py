import numpy as np
import os

base_dir = "dataset/lstm/train/B/1"

for frame_num in range(30):
    file_path = os.path.join(base_dir, f"{frame_num}.npy")
    if os.path.exists(file_path):
        data = np.load(file_path)
        print(f"Frame {frame_num:02d} - shape: {data.shape}")
        print(data)
        print("=" * 40)
    else:
        print(f"Arquivo não encontrado: {file_path}")