import librosa
import soundfile as sf
import os

# This file splits the audio file into 1-seconds segment for processing later
AUDIO_FILE = "../data/comba-1.wav"
OUTPUT_DIR = "../data/segments/" 

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the audio file

y, sr = librosa.load(AUDIO_FILE, sr=16000)

# Split the adudio in fragments of 1 seconds

segment_length = sr #1 second
num_segments = len(y) // segment_length

for i in range(num_segments):
    start = i * segment_length
    end = start + segment_length
    segment = y[start:end]

    output_path = os.path.join(OUTPUT_DIR, f"segment_{i+1}.wav")
    sf.write(output_path, segment, sr)
    print(f"Saved segment: {output_path}")