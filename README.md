# Jump Detection ML 🎧🤸‍♂️

This project was created to **test the feasibility of a mobile app idea**:  
an application capable of **counting jump rope repetitions using only the device’s microphone**, without the need for wearables or additional sensors.  

The repository contains the data, scripts, and models used to experiment with audio processing and machine learning for **jump detection through sound analysis**.

---

## 📁 Project Structure

```
JUMP-DETECTION-ML/
│
├── data/                         # Contains all input and processed audio data
│   ├── augmented_data/          # Artificially generated audio segments (data augmentation)
│   ├── filtered_segments/       # Filtered segments for training
│   ├── jump_segments/           # Segments labeled or detected as jumps
│   ├── noise_samples/           # Noise-only samples to improve robustness
│   ├── segments/                # General segments from raw audio
│   ├── comba-1.wav              # Main raw audio file to be analyzed
│   └── jump_data.csv            # Structured dataset with timestamps and labels
│
├── models/                      # Trained models
│   └── jump_detection_model.pth # PyTorch model for jump detection
│
├── scripts/                     # Data processing, training, and model conversion scripts
│   ├── add_noise.py             # Adds noise to audio segments (augmentation)
│   ├── convert_to_onnx.py       # Converts PyTorch model to ONNX format
│   ├── detect_jumps.py          # Detects jumps in .wav files using trained model
│   ├── filter_audio.py          # Reduces noise or trims silence from audio
│   ├── preprocess_data.py       # Runs segmentation, normalization, feature extraction
│   ├── process_pipeline.py      # Orchestrates full data processing pipeline
│   ├── split_audio.py           # Splits raw audio into short segments
│   └── train_model.py           # Trains the jump detection model

