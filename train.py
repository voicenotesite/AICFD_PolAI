# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1) Definicja modelu (tak jak w AICFD Pro)
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
        # Xavier init + mały bias
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.01)

    def forward(self, x):
        return self.layers(x)


def main():
    # 2) Generowanie syntetycznego zestawu danych
    num_samples = 2000
    # wejścia: punkty (x,y,z) w [-1,1]
    X = np.random.uniform(-1, 1, size=(num_samples, 3)).astype(np.float32)
    # etykiety: Cd = x^2 + y^2 + z^2 + szum gaussowski
    noise = np.random.normal(0, 0.02, size=(num_samples,))
    y = (np.sum(X**2, axis=1) + noise).astype(np.float32).reshape(-1, 1)

    # konwersja do Tensora
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)

    # 3) Przygotowanie treningu
    model = DragPredictor3D()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 4) Pętla treningowa
    num_epochs = 50
    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        preds = model(X_tensor)
        loss = criterion(preds, y_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs}  Loss: {loss.item():.6f}")

    # 5) Zapis wag
    weights_path = "model_weights.pth"
    torch.save(model.state_dict(), weights_path)
    print(f"==> Wytrenowane wagi zapisano w: {weights_path}")


if __name__ == "__main__":
    main()
