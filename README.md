# 🪢 Proyecto de Detección de Saltos con Audio (Versión Simplificada)

Este proyecto permite entrenar un modelo de IA para detectar saltos a la comba a partir de audio, utilizando un enfoque manual y directo para construir el dataset.

---

## 📁 Estructura de carpetas

```
/data
  ├── raw/
  │   ├── jumps/            # Audios donde SOLO hay saltos
  │   └── noise/            # Audios donde NO hay saltos (ruido de gimnasio, etc.)
  ├── classified/
  │   ├── jump_segments/         # Fragmentos con salto (generados automáticamente)
  │   └── non_jump_segments/     # Fragmentos sin salto
  ├── jump_data.csv         # CSV con los MFCCs y etiquetas
/models
  ├── jump_detection_model.pth   # Modelo entrenado
  └── jump_detection_model.onnx  # Versión exportada a ONNX (opcional)
```

---

## 🪜 Flujo de trabajo

### 1. 🎙️ Graba tus audios
Coloca archivos `.wav` en las carpetas correspondientes:
- `data/raw/jumps/` → grabaciones con saltos a la comba
- `data/raw/noise/` → grabaciones sin saltos (ruido ambiente)

---

### 2. ✂️ Ejecuta `split_audio.py`
Este script cortará cada audio en fragmentos de 1 segundo y los clasificará automáticamente como salto o no salto según la carpeta de origen.

---

### 3. 🧠 Ejecuta `preprocess_data.py`
Este script extrae MFCCs (media y desviación estándar) de cada fragmento y genera `jump_data.csv` para entrenar.

---

### 4. 🔬 Ejecuta `train_model.py`
Entrena el modelo de detección de saltos. Guarda el modelo entrenado en `models/jump_detection_model.pth`.

---

### 5. (Opcional) 🤖 Ejecuta `detect_jumps.py`
Clasifica nuevos fragmentos desconocidos usando el modelo entrenado (debes preparar la carpeta `data/filtered_segments/` con nuevos clips).

---

### 6. (Opcional) 📦 Ejecuta `convert_to_onnx.py`
Convierte el modelo a formato ONNX si necesitas exportarlo fuera de Python.

---

## ✅ Requisitos

Instala las dependencias necesarias con:

```bash
pip install librosa torch pandas scikit-learn soundfile
```

---

## 📌 Notas

- El modelo usa 26 características por clip: 13 MFCC medios + 13 desviaciones estándar.
- Se recomienda tener al menos 100 fragmentos por clase para empezar a obtener resultados fiables.