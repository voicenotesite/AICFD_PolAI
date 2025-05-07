import sys
import numpy as np
import torch
import torch.nn as nn
import json
from PyQt5.QtWidgets import (
    QApplication, QVBoxLayout, QWidget, QLabel,
    QPushButton, QFileDialog, QMessageBox, QTextEdit, QComboBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
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
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant_(layer.bias, 0.01)

    def forward(self, x):
        if torch.isnan(x).any():
            return torch.ones((x.shape[0], 1)) * 0.5
        return self.net(x)


# === Główne GUI aplikacji ===
class CFDApp(QWidget):
    def __init__(self):
        super().__init__()
        self.vertices = np.zeros((0, 3))
        self.drag_coeff = 0.0
        self.selected_material = "stal"

        self.init_ui()  # <- najpierw GUI, żeby istniał log_output
        self.materials = self.load_materials()
        self.material_selector.addItems(self.materials.keys())  # <- dodajemy dopiero teraz
        self.init_model()

    def init_ui(self):
        self.setWindowTitle("AICFD Pro v4.0")
        self.setGeometry(100, 100, 1300, 950)
        try:
            self.setWindowIcon(QIcon("icon.png"))
        except Exception:
            pass

        layout = QVBoxLayout()

        self.lbl_status = QLabel("🔵 Status: Gotowy")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("font-size: 18px; font-weight: bold; color: #00ff99;")

        self.btn_load = QPushButton("📁 Wczytaj model STL")
        self.btn_load.clicked.connect(self.load_model)

        self.material_selector = QComboBox()
        self.material_selector.currentTextChanged.connect(self.update_material)

        self.viewer = gl.GLViewWidget()
        self.viewer.setMinimumSize(800, 600)
        self.viewer.setCameraPosition(distance=3)
        self.viewer.opts['distance'] = 5
        self.viewer.setBackgroundColor('k')

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

        layout.addWidget(self.lbl_status)
        layout.addWidget(self.btn_load)
        layout.addWidget(self.material_selector)
        layout.addWidget(self.viewer)
        layout.addWidget(self.log_output)
        self.setLayout(layout)

    def log_message(self, message):
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_output.append(f"[{timestamp}] {message}")
        self.log_output.verticalScrollBar().setValue(
            self.log_output.verticalScrollBar().maximum()
        )

    def load_materials(self):
        try:
            with open("materials.json", "r", encoding="utf-8") as f:
                materials = json.load(f)
            self.log_message("✅ Dane o materiałach wczytane!")
            return materials
        except FileNotFoundError:
            self.log_message("⚠️ Plik materials.json nie istnieje!")
            return {}

    def update_material(self, material):
        self.selected_material = material
        self.log_message(f"🔹 Wybrano materiał: {material}")

        # Jeśli plik STL już jest wczytany, przelicz Cd
        if self.vertices.size > 0:
            self.recalculate_cd()

    def load_model(self):
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

            stl_mesh = mesh.Mesh.from_file(filename)
            vertices = stl_mesh.vectors.reshape(-1, 3)
            if len(vertices) < 4:
                raise ValueError("⚠️ Model musi zawierać co najmniej 4 wierzchołki")

            self.vertices = vertices
            self.update_visualization_mesh(stl_mesh)

            # Po załadowaniu pliku STL przelicz Cd
            self.recalculate_cd()

        except Exception as e:
            self.log_message(f"❌ Błąd: {e}")
            QMessageBox.critical(self, "Błąd", f"Nie można wczytać pliku:\n{e}")

    def recalculate_cd(self):
        material_data = self.materials.get(self.selected_material, {})
        features = self.extract_features(self.vertices)
        input_tensor = torch.FloatTensor(features).unsqueeze(0)

        # Użyj danych materiału do przewidywania Cd
        input_tensor = self.add_material_data_to_features(input_tensor, material_data)

        with torch.no_grad():
            self.drag_coeff = self.model(input_tensor).item()
            self.drag_coeff = max(0.1, min(self.drag_coeff, 2.0))

        self.lbl_status.setText(f"🟢 Cd: {self.drag_coeff:.4f}")
        self.log_message(f"✅ Cd: {self.drag_coeff:.4f}")

    def add_material_data_to_features(self, input_tensor, material_data):
        # Dodaj dane materiału do cech (np. gęstość, lepkość, itd.)
        material_features = np.array([material_data.get("density", 1000), material_data.get("viscosity", 0.001)],
                                     dtype=np.float32)
        return torch.cat((input_tensor, torch.FloatTensor(material_features).unsqueeze(0)), dim=1)

    def update_visualization_mesh(self, stl_mesh):
        self.viewer.clear()
        faces = stl_mesh.vectors
        mesh_data = gl.MeshData(vertexes=faces.reshape(-1, 3), faces=np.arange(len(faces) * 3).reshape(-1, 3))
        mesh_item = gl.GLMeshItem(meshdata=mesh_data, smooth=False, drawEdges=True, edgeColor=(1, 1, 1, 1),
                                  color=(0.5, 0.5, 1, 0.7))
        self.viewer.addItem(mesh_item)

    def extract_features(self, vertices):
        center = np.mean(vertices, axis=0)
        centered = vertices - center
        max_dims = np.max(centered, axis=0)
        min_dims = np.min(centered, axis=0)
        bbox = max_dims - min_dims
        volume = np.abs(np.prod(bbox))
        surface_area = np.linalg.norm(np.cross(centered[1] - centered[0], centered[2] - centered[0]))
        aspect_ratio = bbox[0] / (bbox[1] + 1e-6)
        return np.array([
            np.mean(vertices[:, 0]),
            np.mean(vertices[:, 1]),
            np.mean(vertices[:, 2]),
            np.max(vertices[:, 0]) - np.min(vertices[:, 0]),
            np.max(vertices[:, 1]) - np.min(vertices[:, 1]),
            np.max(vertices[:, 2]) - np.min(vertices[:, 2]),
            volume,
            surface_area,
            aspect_ratio
        ], dtype=np.float32)

    def init_model(self):
        self.model = EnhancedDragPredictor(input_size=9 + 2)  # Dodatkowe dane materiału
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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = CFDApp()
    window.show()
    sys.exit(app.exec_())
