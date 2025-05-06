import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.optimize import minimize

#--------------------------------------------------------------#
# 1.Generowanie danych(parametry NACA + symulacja "Zastępcza")
#--------------------------------------------------------------#
def naca4(number, n_points=100):
    m = int(number[0]) / 100
    p = int(number[1]) / 10
    t = int(number[2:]) / 100

    x = np.linspace(0, 1, n_points)
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    return x, yt
def compute_drag(naca_number, angle_of_attack=0):
    _, yt = naca4(naca_number)
    thickness = np.max(yt)
    drag = thickness * (1+ 0.1 * angle_of_attack**2)
    return drag
np.random.seed(42)
naca_numbers = [f"{np.random.randint(0, 10)}{np.random.randint(0, 10)}{np.random.randint(10, 30)}" for _ in range(500)]
angles = np.random.uniform(-5, 5, 500)

x = np.array([[int(n[0]), int(n[1]), int(n[2:]), a] for n, a in zip(naca_numbers, angles)])
y = np.array([compute_drag(n, a) for n, a in zip(naca_numbers, angles)])

x = (x - x.mean(axis=0)) / x.std(axis=0)
y = (y - y.mean()) / y.std()

split = int(0.8 * len(x))
x_train, x_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]

x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train).view(-1, 1)
x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(y_test).view(-1, 1)

#-------------------------#
# 2.Model sieci neuronowej
#-------------------------#
class DragPredictor(nn.Module):
    def __init__(self, input_size=4):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.layers(x)

model = DragPredictor()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),   lr=0.01)

epochs = 1000
losses = []

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, x_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}:.4f")

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Los (MSE)")
plt.title("Training Progress")
plt.show()

#----------------------------#
# 3.Testowanie i wizualizacja
#----------------------------#

model.eval()
with torch.no_grad():
    predoctions = model(x_test)
    test_loss = criterion(predoctions, y_test)
    print(f"Test Loss: {test_loss.item():.4f}")
plt.scatter(y_test.numpy(), predoctions.numpy(), alpha=0.5)
plt.xlabel("True Drag(Normalized)")
plt.ylabel("Predicted Drag(Normalized)")
plt.title("AI vs. True Drag(Normalized)")
plt.plot([-2, 2], [-2, 2], 'r--')
plt.show()

sample_input = torch.FloatTensor([[0, 4, 12, 2.5]])
predicted_drag = model(sample_input).item()
print(f"Przewidywany opór dla NACA 0412 @ 2.5°: {predicted_drag:.4f} (Znormalizowany")

