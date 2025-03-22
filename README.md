📁 Project Structure
data/
Contains all input and processed audio data.

augmented_data/: Artificially generated audio segments, likely used for data augmentation.

filtered_segments/: Filtered segments, possibly to remove noise or select only relevant ones for training.

jump_segments/: Audio clips containing detected or labeled jumps.

noise_samples/: Noise samples that can be used to train the model to distinguish between jumps and background noise.

segments/: General audio segments, likely the original extracted portions before classification.

comba-1.wav: Main raw audio file to be analyzed.

jump_data.csv: Structured dataset with relevant information such as timestamps, labels, etc.

models/
Contains the trained models.

jump_detection_model.pth: Trained model in PyTorch format. Used to detect jumps from segmented audio.

scripts/
Scripts for processing data, training, and model conversion.

add_noise.py
Adds noise to audio segments to improve model robustness (data augmentation).

convert_to_onnx.py
Converts the PyTorch .pth model to ONNX format for cross-platform deployment and optimization.

detect_jumps.py
Uses the trained model to detect jumps in a full audio file.
Input: .wav file – Output: segments with detected jumps.

filter_audio.py
Filters audio, likely reducing noise, trimming silence, or applying transformations to ease segmentation.

preprocess_data.py
General preprocessing script that combines several steps such as segmentation, normalization, and feature extraction.

process_pipeline.py
Orchestrates the entire pipeline from raw audio input to ready-to-train or predict data.

split_audio.py
Splits raw audio into smaller time windows (segments), which are then labeled and used for training.

train_model.py
Trains the jump detection model using the segmented and preprocessed data.