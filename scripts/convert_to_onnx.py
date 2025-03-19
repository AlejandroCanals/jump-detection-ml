import torch
import torch.onnx
from train_model import JumpDetectionModel

# This file convert the model trained on pyTorch to ONNX format

# Load the model
model = JumpDetectionModel()
model.load_state_dict(torch.load("models/jump_detection_model.pth"))
model.eval()

# Define a test input tensor (with 13 features)
dummy_input = torch.randn(1, 13)

# Export the model to ONNX
onnx_model_path = "models/jump_detection_model.onnx"
torch.onnx.export(model, dummy_input, onnx_model_path, input_names=["input"], output_names=["output"])

print(f"Modelo convertido a ONNX en {onnx_model_path}")
