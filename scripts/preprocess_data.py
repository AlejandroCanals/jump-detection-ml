import librosa
import numpy as np
import os
import panda as pd 

#audio directory

AUDIO_DIR = "data/"

#Extracting features from audio files

def extract_features(audio_path, label):
    y , sr = librosa.load(audio_path, sr = 16000) #Load audio at 16Khz
    mfcc = librosa.feature.mfcc(y=y , sr=sr, n_mfcc=13) #Extract 13 coefficientes MFCC 
    mffc_mean = np.mean(mfcc, axis=1) #Average of MFCC coeffcients
    return mffc_mean, label

data =  []
for file in os.listdir(AUDIO_DIR):
    if file.endswith(".m4a"):
        label = 1 if "jump" in file else 0
        features, label = extract_features(os.path.join(AUDIO_DIR, file), label)
        data.append(*[features, label])

#Save data to CSV file
df = pd.DataFrame(data)
df.to_csv("data/jump_data.csv", index= False)
print("Data saved to CSV file") 