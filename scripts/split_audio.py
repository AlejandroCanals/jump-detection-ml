import os
import librosa
import soundfile as sf

RAW_BASE = "../data/raw"
OUTPUT_JUMP = "../data/classified/jump_segments/"
OUTPUT_NOISE = "../data/classified/non_jump_segments/"

os.makedirs(OUTPUT_JUMP, exist_ok=True)
os.makedirs(OUTPUT_NOISE, exist_ok=True)

for label in ["jumps", "noise"]:
    folder = os.path.join(RAW_BASE, label)
    for filename in os.listdir(folder):
        if not filename.endswith(".wav"):
            continue

        filepath = os.path.join(folder, filename)
        print(f"✂️ Cortando: {filepath}")

        y, sr = librosa.load(filepath, sr=16000)
        segment_length = sr  # 1 segundo
        num_segments = len(y) // segment_length

        for i in range(num_segments):
            start = i * segment_length
            end = start + segment_length
            segment = y[start:end]

            output_folder = OUTPUT_JUMP if label == "jumps" else OUTPUT_NOISE
            output_name = f"{filename.replace('.wav', '')}_seg_{i+1}.wav"
            output_path = os.path.join(output_folder, output_name)

            sf.write(output_path, segment, sr)

        print(f"✅ {num_segments} segmentos guardados en: {output_folder}")
