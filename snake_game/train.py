import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from yarik_topology import ResidualNet
from pathlib import Path

# Імпортуємо шляхи з конфігу (прибрав крапку для прямого запуску)
from config import BIG_MODEL_PATH, DATA_FILE, SMALL_MODEL_PATH, RES_MODEL_PATH

BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
TEST_RATIO = 0.2
SEED = 42

torch.manual_seed(SEED)

# ── Додаткові топології (оновлені до 9 входів) ──────────────────────────────

class SmallNet(nn.Module):
    def __init__(self, input_dim=9, hidden=16, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class BiggerNet(nn.Module):
    def __init__(self, input_dim=9, h1=32, h2=16, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# ── Функції помічники ──────────────────────────────────────────────────────

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
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
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 5 == 0 or epoch == 1:
            test_acc = evaluate(model, test_loader, device)
            print(f"Epoch {epoch:02d} | loss={total_loss:.4f} | test_acc={test_acc:.4f}")

    return model

# ── Main ───────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Завантаження даних з новими колонками (9 фіч + 1 екшн)
    column_names = [
        'dir_up', 'dir_down', 'dir_left', 'dir_right',
        'danger_straight', 'danger_left', 'danger_right',
        'dx_food', 'dy_food', 'action'
    ]

    df = pd.read_csv(DATA_FILE)
    if "action" not in df.columns:
        df.columns = column_names

    print("Dataset size:", len(df))
    print("Action counts:\n", df["action"].value_counts().sort_index())

    X = df.drop(columns=["action"]).values
    y = df["action"].values

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X, y)
    n_test = int(len(dataset) * TEST_RATIO)
    n_train = len(dataset) - n_test

    train_ds, test_ds = random_split(dataset, [n_train, n_test], 
                                     generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Список моделей для тренування
    models_to_train = [
        ("SmallNet", SmallNet(input_dim=9), SMALL_MODEL_PATH),
        ("BiggerNet", BiggerNet(input_dim=9), BIG_MODEL_PATH),
        ("ResidualNet", ResidualNet(input_dim=9, hidden=64), RES_MODEL_PATH)
    ]

    for name, model, save_path in models_to_train:
        print(f"\n=== Training {name} ===")
        print(f"Params: {count_params(model)}")
        
        trained_model = train_model(model, train_loader, test_loader, device, epochs=EPOCHS, lr=LR)
        final_acc = evaluate(trained_model, test_loader, device)
        
        print(f"{name} Final Accuracy: {final_acc:.4f}")
        
        # Зберігаємо ваги
        torch.save(trained_model.state_dict(), save_path)
        print(f"Saved to: {save_path}")

if __name__ == "__main__":
    main()