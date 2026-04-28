"""
ResidualSnakeNet Architecture:
This model uses Skip Connections (Residual Blocks) to create an "information highway."
By adding the input data directly to the output of deeper layers, the network
prevents critical signals (like wall proximity) from being lost during complex
calculations. This results in faster training and more reliable decision-making
compared to standard MLPs.
"""


import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # Перший шар
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        # Другий шар (має повертати таку ж розмірність, як вхід)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # Зберігаємо вхідні дані ("пам'ять")
        identity = x

        # Проходимо крізь шари
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)

        # Головна фішка: додаємо вхід до виходу
        # Це дозволяє сигналу "пролітати" крізь мережу без затухання
        out += identity

        return self.relu(out)


class ResidualNet(nn.Module):
    def __init__(self, input_dim=9, hidden=32, num_classes=3):
        super().__init__()
        # Початкове розширення фіч
        self.input_layer = nn.Linear(input_dim, hidden)

        # Наш блок із залишковим зв'язком
        self.res_block = ResidualBlock(hidden, hidden)

        # Фінальний класифікатор
        self.output_layer = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        x = self.res_block(x)  # Тут працює skip-connection
        x = self.output_layer(x)
        return x
    
def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            preds = torch.argmax(logits, dim=1)

            total += yb.size(0)
            correct += (preds == yb).sum().item()

    return correct / total if total > 0 else 0.0


def train_model(model, train_loader, test_loader, device, epochs=20, lr=1e-3):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:02d} | loss={total_loss:.4f} | test_acc={test_acc:.4f}")

    return model