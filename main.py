import numpy as np
import torch
import torch.nn as nn
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget, QLabel, QPushButton
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph.opengl as gl
from matplotlib.cm import get_cmap
import sys
from stl import mesh  # Do wczytywania modeli 3D


# ----------------------------
# 1. TWOJ MODEL "KARZEŁ" (3D)
# ----------------------------
class DragPredictor3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.layers(x)


model = DragPredictor3D()
model.load_state_dict(torch.load('model_weights.pth'))  # Wczytaj wytrenowane wagi
model.eval()


# ----------------------------
# 2. PROFESJONALNY INTERFEJS 3D
# ----------------------------
class AeroFlowPro(QWidget):
    def __init__(self):
        super().__init__()

        # Konfiguracja okna
        self.setWindowTitle("AeroFlow PRO - Symulator CFD/AI")
        self.setGeometry(100, 100, 1200, 800)

        # Logo i przyciski
        self.logo = QLabel("<h1>AeroFlow™ PRO</h1><p>AI-Powered CFD 3D Simulator</p>")
        self.export_btn = QPushButton("Eksportuj raport (PDF)")
        self.export_btn.clicked.connect(self.export_report)

        # Wizualizacja 3D
        self.viewer = gl.GLViewWidget()
        self.viewer.setCameraPosition(distance=5)

        # Dane 3D (domyślnie kula)
        self.points = self.load_model('sphere.stl')  # Własne modele w .stl
        self.update_visualization()

        # Układ interfejsu
        layout = QVBoxLayout()
        layout.addWidget(self.logo)
        layout.addWidget(self.export_btn)
        layout.addWidget(self.viewer)
        self.setLayout(layout)

        # Timer do symulacji
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self.run_simulation)
        self.sim_timer.start(50)  # 20 FPS

    def load_model(self, filename):
        """Wczytaj model 3D z pliku STL"""
        stl_mesh = mesh.Mesh.from_file(filename)
        return stl_mesh.vectors.reshape(-1, 3)

    def update_visualization(self):
        """Aktualizuj wizualizację na podstawie aktualnego kształtu"""
        with torch.no_grad():
            drag = model(torch.FloatTensor(self.points)).numpy()

        colors = get_cmap('jet')(drag)[:, :3]
        self.scatter = gl.GLScatterPlotItem(
            pos=self.points,
            color=colors,
            size=0.1,
            pxMode=True
        )
        self.viewer.clear()
        self.viewer.addItem(self.scatter)

    def run_simulation(self):
        """Symulacja przepływu w czasie rzeczywistym"""
        # Prosty solver CFD (można zastąpić OpenFOAM)
        self.points[:, 0] += 0.01 * np.sin(time.time())  # Efekt przepływu
        self.update_visualization()

    def export_report(self):
        """Generuj raport PDF"""
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages('aeroflow_report.pdf') as pdf:
            # Strona 1: Wizualizacja 3D
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(
                self.points[:, 0],
                self.points[:, 1],
                self.points[:, 2],
                c=model(torch.FloatTensor(self.points)).numpy(),
                cmap='jet'
            )
            ax.set_title('Rozkład oporu aerodynamicznego')
            pdf.savefig(fig)

            # Strona 2: Dane liczbowe
            plt.figure()
            plt.text(0.1, 0.5,
                     f"Średni opór: {np.mean(drag):.4f}\n"
                     f"Maksymalny opór: {np.max(drag):.4f}\n"
                     f"Minimalny opór: {np.min(drag):.4f}")
            pdf.savefig()
            plt.close()


# ----------------------------
# 3. URUCHOMIENIE APLIKACJI
# ----------------------------
if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Zabezpieczenie licencyjne
    license_check = LicenseValidator.check_license()
    if not license_check:
        print("Brak licencji! Odwiedź www.aeroflow.ai")
        sys.exit(1)

    window = AeroFlowPro()
    window.show()
    sys.exit(app.exec_())