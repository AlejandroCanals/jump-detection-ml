import subprocess

# Lista de scripts a ejecutar en orden
scripts = [
    "split_audio.py",     # 1️⃣ Divide la sesión en fragmentos pequeños
    "filter_audio.py",    # 2️⃣ Filtra música y ruidos irrelevantes
    "detect_jumps.py",    # 3️⃣ Detecta qué fragmentos contienen saltos
    "add_noise.py",       # 4️⃣ Aumenta el dataset con ruido artificial
    "preprocess_data.py", # 5️⃣ Extrae los MFCCs para el entrenamiento
    "train_model.py"      # 6️⃣ Entrena el modelo con los datos procesados
]

print("🚀 Starting the full pipeline process...\n")

for script in scripts:
    print(f"▶ Running {script}...")
    result = subprocess.run(["python", f"scripts/{script}"], capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {script} completed successfully!\n")
    else:
        print(f"❌ ERROR in {script}!\n")
        print(result.stderr)
        break  # Detener la ejecución si hay un error

print("🎯 All steps completed successfully! Model is ready! 🚀")
