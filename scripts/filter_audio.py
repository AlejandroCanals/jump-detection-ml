import librosa
import soundfile as sf
import os

SEGMENTS_DIR = "../data/segments/"
FILTERED_DIR = "../data/filtered_segments/"

os.makedirs(FILTERED_DIR, exist_ok=True)

def filter_frequencies(audio_path, output_path):
    y, sr = librosa.load(audio_path, sr=16000)

    # Aplicar filtro de paso alto para eliminar música y voces
    y_filtered = librosa.effects.preemphasis(y)  

    sf.write(output_path, y_filtered, sr)
    print(f"Filtered audio saved: {output_path}")

for file in os.listdir(SEGMENTS_DIR):
    if file.endswith(".wav"):
        filter_frequencies(os.path.join(SEGMENTS_DIR, file), os.path.join(FILTERED_DIR, file))
