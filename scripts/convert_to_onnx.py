import torch
import torch.onnx
from train_model import JumpDetectionModel

# Este script convierte el modelo entrenado con PyTorch a formato ONNX

# Número de características de entrada (13 MFCC + 13 std = 26)
INPUT_DIM = 26

# Cargar el modelo
model = JumpDetectionModel(input_dim=26)
model.load_state_dict(torch.load("../models/jump_detection_model.pth"))
model.eval()

# Crear una entrada dummy con las nuevas dimensiones
dummy_input = torch.randn(1, INPUT_DIM)

# Ruta de salida ONNX
onnx_model_path = "../models/jump_detection_model.onnx"

# Exportar a ONNX
torch.onnx.export(
    model,
    dummy_input,
    onnx_model_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)

print(f"✅ Modelo convertido a ONNX en: {onnx_model_path}")
