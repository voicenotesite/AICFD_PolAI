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
    def __init__(self, input_size=3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
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
        self.model = DragPredictor3D()

        # Sprawdź czy wagi modelu istnieją
        model_path = "model_weights.pth"
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path))
                self.lbl_status.setText("Status: Model AI załadowany")
            except Exception as e:
                QMessageBox.warning(self, "Błąd", f"Nie można wczytać wag modelu:\n{str(e)}")
                self.lbl_status.setText("Status: Błąd ładowania modelu")
        else:
            QMessageBox.information(self, "Info",
                                    "Plik z wagami modelu nie istnieje.\n"
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

            # Predykcja oporu
            with torch.no_grad():
                drag = self.model(torch.FloatTensor(vertices)).numpy().flatten()

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
