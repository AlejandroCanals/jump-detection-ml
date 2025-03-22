import librosa
import numpy as np
import os

SEGMENTS_DIR = "../data/filtered_segments/"
JUMP_SEGMENTS_DIR = "../data/jump_segments/"

os.makedirs(JUMP_SEGMENTS_DIR, exist_ok=True)

def detect_jumps(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    energy = librosa.feature.rms(y=y)[0]
    threshold = np.percentile(energy, 90)  
    return np.max(energy) > threshold  

for file in os.listdir(SEGMENTS_DIR):
    if file.endswith(".wav"):
        file_path = os.path.join(SEGMENTS_DIR, file)
        if detect_jumps(file_path):
            os.rename(file_path, os.path.join(JUMP_SEGMENTS_DIR, file))
            print(f"Detected jump in: {file}")
