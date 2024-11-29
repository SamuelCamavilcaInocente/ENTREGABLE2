# Importar librerías necesarias
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Dataset
import pandas as pd
import cv2
import numpy as np

# Configuración del dispositivo (CPU o GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================
# 1. Cargar y preprocesar el dataset
# =====================================

class SignLanguageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.labels = self.data.iloc[:, 0].values
        self.images = self.data.iloc[:, 1:].values.reshape(-1, 28, 28).astype('float32')
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Transformaciones: Escalado y normalización
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Cargar dataset
train_data = SignLanguageDataset("sign_mnist_train.csv", transform=transform)
test_data = SignLanguageDataset("sign_mnist_test.csv", transform=transform)

# Dividir dataset en entrenamiento y validación
train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size
train_dataset, val_dataset = random_split(train_data, [train_size, val_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# =====================================
# 2. Definir el modelo
# =====================================
class GestureCNN(nn.Module):
    def __init__(self):
        super(GestureCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 26)  # 26 clases (A-Z)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = GestureCNN().to(device)

# =====================================
# 3. Entrenamiento del modelo
# =====================================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Entrenando el modelo...")
for epoch in range(10):  # Ajusta el número de épocas según lo necesario
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Época {epoch+1}, Pérdida: {running_loss / len(train_loader)}")

    # Evaluar en el conjunto de validación
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Precisión en validación: {100 * correct / total:.2f}%")

# =====================================
# 4. Reconocimiento en tiempo real
# =====================================
print("Activando la cámara para el reconocimiento en tiempo real...")

cap = cv2.VideoCapture(0)
real_time_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

model.eval()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detección de mano y preprocesamiento
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(gray, (28, 28))
    tensor = real_time_transform(roi).unsqueeze(0).to(device)

    # Predicción
    with torch.no_grad():
        output = model(tensor)
        _, predicted = torch.max(output, 1)

    # Mostrar el resultado en la pantalla
    label = chr(predicted.item() + ord('A'))  # Convertir índice a letra
    cv2.putText(frame, f'Gesture: {label}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
