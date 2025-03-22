import os
import torch
import librosa
import numpy as np
from torch import nn

# Rutas
SEGMENTS_DIR = "../data/filtered_segments/"
JUMP_SEGMENTS_DIR = "../data/jump_segments/"
NON_JUMP_SEGMENTS_DIR = "../data/non_jump_segments/"
MODEL_PATH = "../models/jump_detection_model.pth"

# Crear las carpetas si no existen
os.makedirs(JUMP_SEGMENTS_DIR, exist_ok=True)
os.makedirs(NON_JUMP_SEGMENTS_DIR, exist_ok=True)

# 🔁 Esta clase debe coincidir EXACTAMENTE con la de train_model.py
class JumpDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(13, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.activation(self.fc3(x))
        return x

# Cargar el modelo entrenado
model = JumpDetectionModel()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Clasificar los segmentos
for filename in os.listdir(SEGMENTS_DIR):
    if filename.endswith(".wav"):
        filepath = os.path.join(SEGMENTS_DIR, filename)

        # Extraer MFCCs
        y, sr = librosa.load(filepath, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        input_tensor = torch.tensor(mfcc_mean, dtype=torch.float32).unsqueeze(0)

        # Clasificación con el modelo
        with torch.no_grad():
            prediction = model(input_tensor).item()

        # Mover a la carpeta correspondiente
        if prediction > 0.9:
            dest_path = os.path.join(JUMP_SEGMENTS_DIR, filename)
            os.rename(filepath, dest_path)
            print(f"Jump detected in: {filename}")
        else:
            dest_path = os.path.join(NON_JUMP_SEGMENTS_DIR, filename)
            os.rename(filepath, dest_path)
            print(f" No jump in: {filename}")
