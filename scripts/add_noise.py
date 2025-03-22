import librosa
import numpy as np
import soundfile as sf
import os

AUDIO_DIR = "../data/jump_segments/"
NOISE_DIR = "../data/noise_samples/"
AUGMENTED_DIR = "../data/augmented_data/"

os.makedirs(AUGMENTED_DIR, exist_ok=True)

def add_noise(audio_path, noise_path, output_path, noise_level=0.2):
    y, sr = librosa.load(audio_path, sr=16000)
    noise, _ = librosa.load(noise_path, sr=16000)

    if len(noise) > len(y):
        noise = noise[:len(y)]
    else:
        noise = np.pad(noise, (0, len(y) - len(noise)))

    y_noisy = y + noise_level * noise  

    sf.write(output_path, y_noisy, sr)
    print(f"Generated noisy audio: {output_path}")

for file in os.listdir(AUDIO_DIR):
    if file.endswith(".wav"):
        for noise_file in os.listdir(NOISE_DIR):
            if noise_file.endswith(".wav"):
                output_file = f"{AUGMENTED_DIR}/{file.replace('.wav', '')}_noisy_{noise_file}"
                add_noise(os.path.join(AUDIO_DIR, file), os.path.join(NOISE_DIR, noise_file), output_file)
