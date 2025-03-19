import torch
import torch.nn as nn
import numpy as np

# Definir la estructura del modelo nuevamente
class JumpDetectionModel(nn.Module):
    def __init__(self):
        super(JumpDetectionModel, self).__init__()
        self.fc1 = nn.Linear(13, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Cargar el modelo
model = JumpDetectionModel()
model.load_state_dict(torch.load("models/jump_detection_model.pth"))
model.eval()

# Simular una entrada de audio procesada (MFCC)
sample_input = torch.tensor(np.random.rand(1, 13).astype(np.float32))

# Hacer una predicción
with torch.no_grad():  # Evitar que PyTorch calcule gradientes
    prediction = model(sample_input).item()

# Hacer una predicción
# Interpretar la salida (0 = No salto, 1 = Salto)
threshold = 0.5  # Umbral de decisión
prediction_label = "Salto detectado" if prediction > threshold else "No es un salto"

# Mostrar resultado
print(f"Salida del modelo: {prediction:.4f}")
print(f"Interpretación: {prediction_label}")
