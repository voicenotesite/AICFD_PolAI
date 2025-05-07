import sys
import numpy as np
import torch
import torch.nn as nn
from PyQt5.QtWidgets import (
    QApplication, QVBoxLayout, QWidget, QLabel,
    QPushButton, QFileDialog, QMessageBox, QTextEdit
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QVector3D
import pyqtgraph.opengl as gl
import os
from stl import mesh


# === Model AI do przewidywania Cd ===
class EnhancedDragPredictor(nn.Module):
    def __init__(self, input_size=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(128),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        self._init_weights()

    def _init_weights(self):
        """Inicjalizacja wag dla stabilności"""
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant_(layer.bias, 0.01)

    def forward(self, x):
        if torch.isnan(x).any():
            return torch.ones((x.shape[0], 1)) * 0.5  # Wartość domyślna
        return self.net(x)


# === Główne GUI aplikacji ===
class CFDApp(QWidget):
    def __init__(self):
        super().__init__()
        self.vertices = np.zeros((0, 3))
        self.drag_coeff = 0.0
        self.init_ui()
        self.init_model()

    def init_ui(self):
        self.setWindowTitle("AICFD Pro v3.5")
        self.setGeometry(100, 100, 1300, 950)
        try:
            self.setWindowIcon(QIcon("icon.png"))
        except Exception:
            pass

        layout = QVBoxLayout()

        # Panel statusu
        self.lbl_status = QLabel("🔵 Status: Gotowy")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("font-size: 18px; font-weight: bold; color: #00ff99;")

        # Przyciski
        self.btn_load = QPushButton("📁 Wczytaj model STL")
        self.btn_load.clicked.connect(self.load_model)
        self.btn_load.setToolTip("Załaduj plik w formacie STL")

        # Wizualizacja 3D
        self.viewer = gl.GLViewWidget()
        self.viewer.setMinimumSize(800, 600)
        self.viewer.setCameraPosition(distance=3)
        self.viewer.opts['distance'] = 5
        self.viewer.setBackgroundColor('k')

        # Panel logów
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("""
            background-color: #1e1e2d;
            color: #e0e0e0;
            font-family: Consolas;
            font-size: 14px;
            border-radius: 5px;
            padding: 10px;
        """)

        # Układ
        layout.addWidget(self.lbl_status)
        layout.addWidget(self.btn_load)
        layout.addWidget(self.viewer)
        layout.addWidget(self.log_output)
        self.setLayout(layout)

    def log_message(self, message):
        """Dodaje wiadomość do logów z timestampem"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_output.append(f"[{timestamp}] {message}")
        self.log_output.verticalScrollBar().setValue(
            self.log_output.verticalScrollBar().maximum()
        )

    def init_model(self):
        """Inicjalizacja modelu AI"""
        self.model = EnhancedDragPredictor(input_size=9)
        weights_path = "drag_model.pth"
        if os.path.exists(weights_path):
            try:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                state_dict = torch.load(weights_path, map_location=device)
                new_state_dict = {k[4:] if k.startswith('net.') else k: v for k, v in state_dict.items()}
                self.model.load_state_dict(new_state_dict, strict=False)
                self.model.eval()
                self.log_message("✅ Model AI załadowany pomyślnie")
                self.lbl_status.setText("🟢 Status: Model gotowy")
            except Exception as e:
                self.log_message(f"❌ Błąd ładowania wag: {e}")
                self.lbl_status.setText("🔴 Status: Błąd modelu")
        else:
            self.log_message("⚠️ Używany model z domyślnymi wagami")
            self.lbl_status.setText("🟡 Status: Model domyślny")

    def extract_features(self, vertices):
        """Ekstrakcja cech geometrycznych z modelu"""
        centroid = np.mean(vertices, axis=0)
        ptp = np.ptp(vertices, axis=0)
        cov_matrix = np.cov(vertices.T)
        eigenvalues = np.linalg.eigvalsh(cov_matrix)

        # Tablica cech: 3 (centroid) + 3 (rozpiętość) + 3 (statystyki wartości własnych)
        features = np.array([
            *centroid,
            *ptp,
            np.max(eigenvalues),
            np.mean(eigenvalues),
            np.min(eigenvalues)
        ], dtype=np.float32)

        self.log_message(f"📊 Wyekstrahowane cechy: {features}")
        return features

    def load_model(self):
        """Ładowanie pliku STL, renderowanie siatki oraz obliczanie Cd"""
        try:
            filename, _ = QFileDialog.getOpenFileName(
                self,
                "Wybierz plik STL",
                "",
                "Pliki STL (*.stl);;Wszystkie pliki (*)"
            )
            if not filename:
                return
            self.log_message(f"📂 Ładowanie pliku: {filename}")

            # Wczytanie STL
            stl_mesh = mesh.Mesh.from_file(filename)
            vertices = stl_mesh.vectors.reshape(-1, 3)
            if len(vertices) < 4:
                raise ValueError("⚠️ Model musi zawierać co najmniej 4 wierzchołki")

            self.vertices = vertices

            # Aktualizacja wizualizacji – renderowanie jako siatki 3D
            self.update_visualization_mesh(stl_mesh)

            # Ekstrakcja cech i obliczanie Cd
            features = self.extract_features(vertices)
            input_tensor = torch.FloatTensor(features).unsqueeze(0)
            with torch.no_grad():
                self.drag_coeff = self.model(input_tensor).item()
                self.drag_coeff = max(0.1, min(self.drag_coeff, 2.0))  # Fizyczne ograniczenia

            self.lbl_status.setText(f"🟢 Cd: {self.drag_coeff:.4f}")
            self.log_message(f"✅ Cd: {self.drag_coeff:.4f}")
        except Exception as e:
            self.log_message(f"❌ Błąd: {e}")
            QMessageBox.critical(self, "Błąd", f"Nie można wczytać pliku:\n{e}")

    def update_visualization_mesh(self, stl_mesh):
        """Renderowanie modelu STL jako siatki 3D wraz z dodaniem siatki referencyjnej"""
        try:
            vertices = stl_mesh.vectors.reshape(-1, 3)
            n_triangles = stl_mesh.vectors.shape[0]
            faces = np.arange(vertices.shape[0]).reshape(n_triangles, 3)
            mesh_item = gl.GLMeshItem(
                vertexes=vertices,
                faces=faces,
                drawEdges=True,
                drawFaces=True,
                edgeColor=(1, 1, 1, 1),
                color=(0.5, 0.5, 1, 0.7),
                smooth=False,
                shader='shaded'
            )

            self.viewer.clear()

            # Dodajemy grid referencyjny
            grid = gl.GLGridItem()
            grid.setSize(200, 200)
            grid.setSpacing(10, 10)
            self.viewer.addItem(grid)

            # Dodajemy siatkę modelu
            self.viewer.addItem(mesh_item)

            # Ustawienie kamery – wyśrodkowanie na centroidzie modelu
            center = np.mean(vertices, axis=0)
            center_vec = QVector3D(center[0], center[1], center[2])
            self.viewer.opts['center'] = center_vec
            distance = np.linalg.norm(np.ptp(vertices, axis=0))
            self.viewer.setCameraPosition(distance=distance * 2)
            self.viewer.update()

            self.log_message("✅ Wizualizacja 3D zaktualizowana (mesh).")
        except Exception as e:
            self.log_message(f"❌ Błąd wizualizacji (mesh): {e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CFDApp()
    window.show()
    sys.exit(app.exec_())
