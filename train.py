import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import urllib.request
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import requests
from scipy.spatial.distance import cdist  # Dodany import

# Ustawienie urządzenia (CUDA lub CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Klasa modelu
class AdvancedDragPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(18, 256),  # Zmieniona liczba cech wejściowych na 18
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.Dropout(0.2)
        )
        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.01)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.regressor(features)

# Funkcja do pobierania danych
def download_airfoil_data(url):
    try:
        raw = urllib.request.urlopen(url).read().decode()
        coords = []
        for line in raw.splitlines():
            line = line.strip()
            # Ignorujemy linie z nagłówkami i komentarzami
            if line and not line.lower().startswith(('naca', 'airfoil')) and not any(char.isalpha() for char in line):
                coords.append(list(map(float, line.split())))
        return np.array(coords)
    except Exception as e:
        print(f"⚠️ Nie udało się pobrać danych z {url}: {e}")
        return None

# Funkcja do generowania danych
def generate_advanced_training_data(num_samples=5000):
    shapes = ['sphere', 'cube', 'cylinder']
    uiuc_airfoils = ['e423', 'mh114', 'goe398', 's7055']
    data = []

    for _ in tqdm(range(num_samples), desc="Generowanie syntetycznych danych"):
        shape_type = np.random.choice(shapes)
        if shape_type == 'sphere':
            pts = np.random.randn(100, 3)
            pts /= np.linalg.norm(pts, axis=1)[:, None]
            params = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])  # Nowa liczba parametrów
        elif shape_type == 'cube':
            pts = np.random.rand(100, 3) * 2 - 1
            params = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])  # Nowa liczba parametrów
        elif shape_type == 'cylinder':
            radius = np.random.uniform(0.5, 1.5)
            height = np.random.uniform(1.0, 3.0)
            angle = np.random.rand(100) * 2 * np.pi
            z = np.random.rand(100) * height
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            pts = np.stack([x, y, z], axis=1)
            params = np.array([0.0, 0.0, 1.0, height / radius, 0.0, 0.0, 1.0, 0.0, 0.0])  # Nowa liczba parametrów

        features = extract_geometric_features(pts)
        full_features = np.concatenate([features, params])
        drag = simulate_cfd(pts, params)
        data.append((full_features, drag))

    # Pobranie i dodanie danych z UIUC airfoils
    for airfoil in tqdm(uiuc_airfoils, desc="Pobieranie airfoili UIUC"):
        url = f"https://m-selig.ae.illinois.edu/ads/coord/{airfoil}.dat"
        coords = download_airfoil_data(url)
        if coords is not None:
            pts = np.pad(coords, ((0, 0), (0, 1)), constant_values=0)  # dodanie Z=0
            params = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.5, 1.0, 0.0, 0.0])  # Nowa liczba parametrów
            features = extract_geometric_features(pts)
            full_features = np.concatenate([features, params])
            drag = simulate_cfd(pts, params)
            data.append((full_features, drag))

    X = np.array([d[0] for d in data])
    y = np.array([d[1] for d in data])
    return X, y

# Funkcja do ekstrakcji cech geometrycznych
def extract_geometric_features(points):
    centroid = np.mean(points, axis=0)
    cov_matrix = np.cov(points.T)
    eigenvalues = np.linalg.eigvals(cov_matrix)
    return np.array([
        eigenvalues[0], eigenvalues[1], eigenvalues[2],
        np.max(cdist(points, points)),
        np.mean(points[:, 0]),
        np.mean(points[:, 1]),
        np.mean(points[:, 2]),
        np.std(points[:, 0]),  # Dodano std dla lepszego opisania rozrzutu
        np.std(points[:, 1]),  # Dodano std dla lepszego opisania rozrzutu
    ])

# Funkcja do symulacji CFD
def simulate_cfd(points, params):
    shape_factor = params[0] * 0.8 + params[1] * 1.2 + params[2] * 1.0
    roughness = 1.0 + params[6] * 0.1
    re_effect = np.log(5000 + params[3] * 10000) / 10
    return shape_factor * roughness * re_effect + np.random.normal(0, 0.01)

# Funkcja do trenowania modelu
def train():
    X, y = generate_advanced_training_data(5000)

    # Normalizacja danych
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Tworzenie modelu i optymalizatora
    model = AdvancedDragPredictor().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.MSELoss()

    batch_size = 512
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for i in range(0, len(X), batch_size):
            batch_X = torch.FloatTensor(X[i:i + batch_size]).to(device)
            batch_y = torch.FloatTensor(y[i:i + batch_size]).unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch} | Loss: {epoch_loss / (len(X) / batch_size):.4f}')

    # Zapisz model i dane treningowe
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
    }, 'drag_model.pth')

    print("Model i dane zapisane!")

# Główna funkcja
if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"⚠️ Błąd: {e}")
