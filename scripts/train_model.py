import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import os

# Cargar datos reales desde CSV
df = pd.read_csv("../data/jump_data.csv")
X = df.drop("label", axis=1).values.astype(np.float32)
y = df["label"].values.astype(np.float32)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convertir a tensores
X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)
y_train = torch.tensor(y_train).unsqueeze(1)
y_test = torch.tensor(y_test).unsqueeze(1)

# Modelo mejorado
class JumpDetectionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x

# Entrenamiento
input_dim = X.shape[1]
model = JumpDetectionModel(input_dim=26)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Evaluación
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    predictions_binary = (predictions > 0.5).float()
    accuracy = (predictions_binary == y_test).float().mean()
    print(f"Accuracy on test set: {accuracy.item() * 100:.2f}%")

# Guardar el modelo
os.makedirs("../models", exist_ok=True)
torch.save(model.state_dict(), "../models/jump_detection_model.pth")
print("Modelo guardado correctamente en ../models/jump_detection_model.pth")
