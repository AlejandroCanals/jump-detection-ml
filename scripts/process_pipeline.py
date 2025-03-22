import subprocess
import sys

# List of scripts to run in order
scripts = [
    "split_audio.py",     # 1️⃣ Splits the session into small segments
    "filter_audio.py",    # 2️⃣ Filters out music and irrelevant noise
    "add_noise.py",       # 3️⃣ Augments the dataset with artificial noise
    "detect_jumps.py",    # 4️⃣ Detects which segments contain jumps
    "preprocess_data.py", # 5️⃣ Extracts MFCCs for model training
    "train_model.py"      # 6️⃣ Trains the model with the processed data
]

print("🚀 Starting the full pipeline process...\n")

for script in scripts:
    print(f"▶ Running {script}...")
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {script} completed successfully!\n")
    else:
        print(f"❌ ERROR in {script}!\n")
        print(result.stderr)
        break 

print("🎯 All steps completed successfully! Model is ready! 🚀")
