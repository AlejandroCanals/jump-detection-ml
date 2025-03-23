import os
import librosa
import numpy as np
import pandas as pd

# Carpetas de entrada
JUMP_DIR = "../data/jump_segments/"
NON_JUMP_DIR = "../data/non_jump_segments/"

# Archivo de salida
OUTPUT_CSV = "../data/jump_data.csv"

# Función para extraer características
def extract_features(audio_path, label):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    features = np.concatenate([mfcc_mean, mfcc_std])
    return np.append(features, label)  # 26 características + etiqueta

data = []

# Procesar segmentos CON salto (label = 1)
for file in os.listdir(JUMP_DIR):
    if file.endswith(".wav"):
        path = os.path.join(JUMP_DIR, file)
        features = extract_features(path, label=1)
        data.append(features)

# Procesar segmentos SIN salto (label = 0)
for file in os.listdir(NON_JUMP_DIR):
    if file.endswith(".wav"):
        path = os.path.join(NON_JUMP_DIR, file)
        features = extract_features(path, label=0)
        data.append(features)

# Guardar CSV
columns = [f"mfcc_{i}" for i in range(13)] + [f"mfcc_std_{i}" for i in range(13)] + ["label"]
df = pd.DataFrame(data, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)

print(f"[OK] CSV generado con {len(df)} ejemplos en: {OUTPUT_CSV}")
print(df['label'].value_counts())
