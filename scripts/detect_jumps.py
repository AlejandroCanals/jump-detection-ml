import os
import torch
import librosa
import numpy as np
from torch import nn

# Carpetas de entrada y salida
SEGMENTS_DIR = "../data/filtered_segments/"
JUMP_SEGMENTS_DIR = "../data/jump_segments/"
NON_JUMP_SEGMENTS_DIR = "../data/non_jump_segments/"
MODEL_PATH = "../models/jump_detection_model.pth"

# Crear carpetas si no existen
os.makedirs(JUMP_SEGMENTS_DIR, exist_ok=True)
os.makedirs(NON_JUMP_SEGMENTS_DIR, exist_ok=True)

# Modelo actualizado
class JumpDetectionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.activation(self.fc3(x))
        return x

# Obtener dimensión de entrada desde un ejemplo
sample_file = next((f for f in os.listdir(SEGMENTS_DIR) if f.endswith(".wav")), None)
if not sample_file:
    raise FileNotFoundError("No se encontraron archivos de audio en filtered_segments.")

y, sr = librosa.load(os.path.join(SEGMENTS_DIR, sample_file), sr=16000)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
mfcc_mean = np.mean(mfcc, axis=1)
mfcc_std = np.std(mfcc, axis=1)
features = np.concatenate([mfcc_mean, mfcc_std])
input_dim = len(features)

# Cargar el modelo
model = JumpDetectionModel(input_dim=26)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Umbral de detección (ajustable)
THRESHOLD = 0.7

# Clasificación
for filename in os.listdir(SEGMENTS_DIR):
    if not filename.endswith(".wav"):
        continue

    filepath = os.path.join(SEGMENTS_DIR, filename)

    try:
        y, sr = librosa.load(filepath, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        features = np.concatenate([mfcc_mean, mfcc_std])
        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            prediction = model(input_tensor).item()

        if prediction > THRESHOLD:
            dest_path = os.path.join(JUMP_SEGMENTS_DIR, filename)
            print(f" Salto detectado en: {filename} (score: {prediction:.2f})")
        else:
            dest_path = os.path.join(NON_JUMP_SEGMENTS_DIR, filename)
            print(f" No salto en: {filename} (score: {prediction:.2f})")

        os.rename(filepath, dest_path)

    except Exception as e:
        print(f" Error procesando {filename}: {e}")
