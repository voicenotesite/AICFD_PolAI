import hashlib
import json
import os
import torch
import torch.nn as nn
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QVBoxLayout, QWidget, QLabel,
    QPushButton, QFileDialog, QMessageBox, QTextEdit, QComboBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
import pyqtgraph.opengl as gl
from stl import mesh


# === Model AI do przewidywania Cd ===
class EnhancedDragPredictor(nn.Module):
    def __init__(self, input_size=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),  # Zwiększenie liczby neuronów w pierwszej warstwie
            nn.LeakyReLU(0.1),
            nn.LayerNorm(256),
            nn.Linear(256, 512),  # Zwiększenie liczby neuronów w drugiej warstwie
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(512, 1024),  # Kolejna warstwa dla lepszego dopasowania
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 1)
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
        self.selected_material = "steel"

        self.init_ui()  # Inicjalizowanie UI
        self.materials = self.load_materials()  # Wczytanie materiałów
        self.material_selector.addItems(self.materials.keys())  # Wybór materiału
        self.init_model()  # Inicjalizacja modelu (dodajemy tę metodę)

        # Wczytanie cache wyników obliczeń
        self.results_cache = self.load_results_cache()

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

        if self.vertices.size > 0:
            self.recalculate_cd()

    def load_results_cache(self):
        cache_file = "results_cache.json"
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                self.log_message(f"❌ Błąd ładowania wyników z pamięci podręcznej: {e}")
        return {}

    def save_results_cache(self):
        cache_file = "results_cache.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(self.results_cache, f, indent=4)
        except Exception as e:
            self.log_message(f"❌ Błąd zapisywania wyników do pamięci podręcznej: {e}")

    def calculate_hash(self, filename):
        """Oblicz hash pliku STL, aby unikalnie identyfikować plik."""
        hash_sha256 = hashlib.sha256()
        try:
            with open(filename, "rb") as f:
                while chunk := f.read(8192):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            self.log_message(f"❌ Błąd obliczania hasha pliku STL: {e}")
            return None

    def recalculate_cd(self):
        file_hash = self.calculate_hash(self.current_file)

        # Sprawdzanie cache
        if file_hash in self.results_cache:
            self.drag_coeff = self.results_cache[file_hash]
            self.log_message(f"✅ Użyto zapisanej wartości Cd: {self.drag_coeff:.4f}")
            self.lbl_status.setText(f"🟢 Cd: {self.drag_coeff:.4f}")
            return

        material_data = self.materials.get(self.selected_material, {})
        features = self.extract_features(self.vertices)
        input_tensor = torch.FloatTensor(features).unsqueeze(0)

        # Użyj danych materiału do przewidywania Cd
        input_tensor = self.add_material_data_to_features(input_tensor, material_data)

        with torch.no_grad():
            self.drag_coeff = self.model(input_tensor).item()

        self.drag_coeff = validate_cd(self.drag_coeff)

        if self.drag_coeff >= 0.1 and self.drag_coeff <= 2.0:
            self.lbl_status.setText(f"🟢 Cd: {self.drag_coeff:.4f}")
            self.log_message(f"✅ Cd: {self.drag_coeff:.4f}")
        else:
            self.lbl_status.setText("🔴 Cd: Wynik nierealistyczny!")
            self.log_message("❌ Błąd: Nierealistyczny wynik Cd.")

        # Zapisz wynik w cache
        if file_hash:
            self.results_cache[file_hash] = self.drag_coeff
            self.save_results_cache()

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

            # Zapisywanie ścieżki pliku
            self.current_file = filename

            # Po załadowaniu pliku STL przelicz Cd
            self.recalculate_cd()

        except Exception as e:
            self.log_message(f"❌ Błąd: {e}")
            QMessageBox.critical(self, "Błąd", f"Nie można wczytać pliku:\n{e}")

    def init_model(self):
        """ Inicjalizowanie modelu AI do przewidywania Cd """
        self.model = EnhancedDragPredictor(input_size=9)  # Tworzenie instancji modelu AI
        self.model.eval()  # Ustawienie modelu w tryb ewaluacji (nie trenujemy modelu)

    def update_visualization_mesh(self, stl_mesh):
        """Metoda aktualizująca wizualizację modelu 3D w PyQt"""
        # Zbieramy dane wierzchołków i trójkątów z pliku STL
        vertices = stl_mesh.vectors.reshape(-1, 3)
        faces = np.array([[i, i + 1, i + 2] for i in range(0, len(vertices), 3)])

        # Stworzenie obiektu 3D
        mesh_item = gl.GLMeshItem(vertexes=vertices, faces=faces, color=(0.8, 0.8, 0.8, 1.0), shader="phong")
        self.viewer.addItem(mesh_item)


# === Uruchomienie aplikacji ===
if __name__ == "__main__":
    app = QApplication([])  # Tworzymy aplikację Qt
    window = CFDApp()  # Tworzymy główne okno aplikacji
    window.show()  # Wyświetlamy okno aplikacji
    app.exec_()
