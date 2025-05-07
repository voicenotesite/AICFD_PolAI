import numpy as np
import torch
import torch.nn as nn
from PyQt5.QtWidgets import (QApplication, QVBoxLayout,
                             QWidget, QLabel, QPushButton,
                             QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
import pyqtgraph.opengl as gl
import matplotlib.pyplot as plt
import sys
import os

# 1. SPRAWDZENIE CZY numpy-stl JEST ZAINSTALOWANE
try:
    from stl import mesh
except ImportError:
    print("BŁĄD: Zainstaluj wymaganą bibliotekę komendą:")
    print("pip install numpy-stl")
    sys.exit(1)

# 2. MODEL AI Z ZABEZPIECZENIAMI
class DragPredictor3D(nn.Module):
    def __init__(self, input_size=9):  # Zmieniono na 9, bo model trenowany na 9 cechach
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.SiLU(),
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.Linear(512, 1)
        )
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.01)

    def forward(self, x):
        if torch.isnan(x).any():
            return torch.zeros((x.shape[0], 1))
        return self.layers(x)

# 3. GŁÓWNA KLASA APLIKACJI
class CFDApp(QWidget):
    def __init__(self):
        super().__init__()
        self.vertices = np.zeros((0, 3))
        self.drag = 1.0
        self.init_ui()
        self.init_model()

    def init_ui(self):
        self.setWindowTitle("AICFD Pro")
        self.setGeometry(100, 100, 800, 600)
        self.setWindowIcon(QIcon("icon.png"))  # Ustawienie ikony aplikacji

        layout = QVBoxLayout()

        # Przyciski
        self.btn_load = QPushButton("Wczytaj model STL")
        self.btn_load.clicked.connect(self.load_model)

        # Etykiety
        self.lbl_status = QLabel("Status: Gotowy")
        self.lbl_status.setAlignment(Qt.AlignCenter)

        # Wizualizacja 3D
        self.viewer = gl.GLViewWidget()
        self.viewer.setCameraPosition(distance=5)

        layout.addWidget(self.lbl_status)
        layout.addWidget(self.btn_load)
        layout.addWidget(self.viewer)
        self.setLayout(layout)

        # Stylizacja
        self.setStyleSheet("""
            QWidget {
                background-color: #2c3e50;
                color: #ecf0f1;
                font-family: Arial;
                font-size: 14px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)

    def normalize_vertices(self, vertices):
        vertices = vertices - vertices.mean(axis=0)
        return vertices / (np.max(np.linalg.norm(vertices, axis=1)) + 1e-8)

    def init_model(self):
        """Inicjalizacja modelu z zabezpieczeniami"""
        self.model = DragPredictor3D(input_size=9)  # Zmieniono na 9 cech wejściowych

        # Sprawdź czy wagi modelu istnieją
        drag_model_path = "drag_model.pth"
        drag_predictor_path = "drag_predictor.pth"

        # Wczytanie wag z odpowiedniego pliku
        if os.path.exists(drag_model_path):
            try:
                self.model.load_state_dict(torch.load(drag_model_path))
                self.lbl_status.setText("Status: Załadowano wagi z drag_model.pth")
            except Exception as e:
                QMessageBox.warning(self, "Błąd", f"Nie można wczytać wag modelu z drag_model.pth:\n{str(e)}")
                self.lbl_status.setText("Status: Błąd ładowania wag drag_model.pth")
        elif os.path.exists(drag_predictor_path):
            try:
                self.model.load_state_dict(torch.load(drag_predictor_path))
                self.lbl_status.setText("Status: Załadowano wagi z drag_predictor.pth")
            except Exception as e:
                QMessageBox.warning(self, "Błąd", f"Nie można wczytać wag modelu z drag_predictor.pth:\n{str(e)}")
                self.lbl_status.setText("Status: Błąd ładowania wag drag_predictor.pth")
        else:
            QMessageBox.information(self, "Info",
                                    "Brak pliku wag.\n"
                                    "Zostanie użyty model z losowymi wagami.")
            self.lbl_status.setText("Status: Model z losowymi wagami")

    def load_model(self):
        """Wczytaj plik STL z pełną obsługą błędów"""
        try:
            filename, _ = QFileDialog.getOpenFileName(
                self, "Otwórz plik STL", "", "Pliki STL (*.stl)"
            )

            if not filename:
                return

            if not os.path.exists(filename):
                raise FileNotFoundError(f"Plik {filename} nie istnieje")

            # Wczytanie siatki 3D
            stl_mesh = mesh.Mesh.from_file(filename)
            vertices = stl_mesh.vectors.reshape(-1, 3)

            # Normalizacja wierzchołków
            vertices = self.normalize_vertices(vertices)

            # Extrakcja cech geometrycznych (9 cech jak w modelu)
            features = self.extract_features(vertices)

            # Predykcja oporu
            with torch.no_grad():
                drag = self.model(torch.FloatTensor(features).unsqueeze(0)).numpy().flatten()

            # Wizualizacja
            colors = plt.get_cmap('viridis')(
                (drag - drag.min()) / (drag.max() - drag.min())
            )

            self.viewer.clear()
            self.viewer.addItem(
                gl.GLScatterPlotItem(
                    pos=vertices,
                    color=colors,
                    size=0.1,
                    pxMode=True
                )
            )

            self.lbl_status.setText(
                f"Status: Załadowano {len(vertices)} wierzchołków | "
                f"Śr. opór: {drag.mean():.4f}"
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Krytyczny błąd",
                f"Nie można wczytać modelu:\n{str(e)}"
            )
            self.lbl_status.setText("Status: Błąd wczytywania")

    def extract_features(self, vertices):
        # Przykładowa funkcja do ekstrakcji 9 cech (można ją dostosować)
        centroid = np.mean(vertices, axis=0)
        max_distance = np.max(np.linalg.norm(vertices - centroid, axis=1))
        return np.concatenate([
            centroid,
            [max_distance]
        ])  # Zwróć 9 cech, tutaj tylko przykładowo

    def update_visualization(self):
        if len(self.vertices) == 0:
            return
        colors = plt.get_cmap('viridis')(
            (self.drag - 0.1) / (2.0 - 0.1)
        )
        self.viewer.clear()
        self.viewer.addItem(
            gl.GLScatterPlotItem(
                pos=self.vertices,
                color=colors,
                size=0.1,
                pxMode=True
            )
        )


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CFDApp()
    window.show()
    sys.exit(app.exec_())
