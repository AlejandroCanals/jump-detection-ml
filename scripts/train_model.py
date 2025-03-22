import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split

# This script trains a jump detection model using a neural network in PyTorch,  
# evaluates its accuracy, and saves the trained model for later use.

# Simulate data if no dataset is available
num_samples = 100
num_features = 13  # Like MFCC coefficients
X = np.random.rand(num_samples, num_features).astype(np.float32)
y = np.random.randint(0, 2, num_samples).astype(np.float32)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Convert data to PyTorch tensors
X_train, X_test = torch.tensor(X_train), torch.tensor(X_test)
y_train, y_test = torch.tensor(y_train).unsqueeze(1), torch.tensor(y_test).unsqueeze(1)

# Define the jump detection model
class JumpDetectionModel(nn.Module):
    def __init__(self):
        super(JumpDetectionModel, self).__init__()
        self.fc1 = nn.Linear(num_features, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Initialize the model, loss function, and optimizer
model = JumpDetectionModel()
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(50):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Evaluate the model on test data (without computing gradients)
with torch.no_grad():
    predictions = model(X_test)
    predictions = (predictions > 0.5).float()  # Convert probabilities to 0 or 1
    accuracy = (predictions == y_test).float().mean()
    print(f"Model accuracy: {accuracy.item() * 100:.2f}%")

# Save the model
torch.save(model.state_dict(), "../models/jump_detection_model.pth")
print("Model saved to models/jump_detection_model.pth")
