import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# === Model AI do przewidywania Cd ===
class EnhancedDragPredictor(nn.Module):
    def __init__(self, input_size=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(0.1),  # Stabilniejsza aktywacja
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
        return self.net(x)


# === Funkcja generująca dane treningowe ===
def generate_data(num_samples, input_dim):
    """
    Generuje losowe dane wejściowe X oraz odpowiadające im wartości Cd.
    Modeluje się je jako:
       y = 0.5 + 2.0*(0.6*X0 + 0.3*X1 + 0.1*X2) + szum
    Dzięki temu uzyskamy wartości Cd w przybliżeniu w zakresie 0.5 - 2.1.
    """
    X = np.random.rand(num_samples, input_dim)
    y = 0.5 + 2.0 * (0.6 * X[:, 0] + 0.3 * X[:, 1] + 0.1 * X[:, 2])
    noise = 0.1 * np.random.randn(num_samples)  # Drobny szum
    y = y + noise

    # Konwersja do tensorów: normalizujemy tylko cechy wejściowe
    X_tensor = torch.tensor(X, dtype=torch.float32)
    X_tensor = (X_tensor - X_tensor.mean(dim=0)) / (X_tensor.std(dim=0) + 1e-8)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    return X_tensor, y_tensor


# === Proces treningowy ===
def train(model, X, y, epochs=500, batch_size=64, lr=1e-3, path="drag_model.pth"):
    dataset = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    loss_fn = nn.MSELoss()

    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x_batch, y_batch in tqdm(dataset, desc=f"Epoka {epoch + 1}/{epochs}", leave=False):
            optimizer.zero_grad()
            output = model(x_batch).flatten()
            loss = loss_fn(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(dataset)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), path)

        if epoch % 50 == 0:
            # Dodatkowe logowanie: przykładowe wyjście pierwszego batcha
            test_output = model(next(iter(dataset))[0]).flatten().detach().cpu().numpy()
            print(f"Epoka {epoch}: Loss = {avg_loss:.4f}, Przykładowe wyjście: {test_output[:5]}")

    print(f"🎯 Najlepszy model zapisany do {path}")


if __name__ == "__main__":
    input_dim = 9
    X_tensor, y_tensor = generate_data(5000, input_dim)
    model = EnhancedDragPredictor(input_size=input_dim)

    # Trening modelu
    train(model, X_tensor, y_tensor, epochs=500, batch_size=64, lr=1e-3)

    # Testowanie modelu na nowych danych
    model.eval()
    X_test, y_test = generate_data(10, input_dim)
    with torch.no_grad():
        predictions = model(X_test).flatten().cpu().numpy()
    print("\nTestowe przewidywania (surowe Cd):", predictions)
