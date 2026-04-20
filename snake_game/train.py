import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from .config import BIG_MODEL_PATH, DATA_FILE, SMALL_MODEL_PATH

BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
TEST_RATIO = 0.2
SEED = 42

torch.manual_seed(SEED)


class SmallNet(nn.Module):
    def __init__(self, input_dim=13, hidden=16, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class BiggerNet(nn.Module):
    def __init__(self, input_dim=13, h1=32, h2=16, num_classes=3):
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


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    df = pd.read_csv(DATA_FILE)
    print("Dataset size:", len(df))
    print(df["action"].value_counts().sort_index())

    X = df.drop(columns=["action"]).values
    y = df["action"].values

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X, y)

    n_total = len(dataset)
    n_test = int(n_total * TEST_RATIO)
    n_train = n_total - n_test

    train_ds, test_ds = random_split(
        dataset,
        [n_train, n_test],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    small = SmallNet()
    big = BiggerNet()

    print("\nSmallNet params:", count_params(small))
    print("BiggerNet params:", count_params(big))

    print("\n=== Training SmallNet ===")
    small = train_model(small, train_loader, test_loader, device, epochs=EPOCHS, lr=LR)
    small_acc = evaluate(small, test_loader, device)
    print(f"SmallNet final test accuracy: {small_acc:.4f}")
    torch.save(small.state_dict(), SMALL_MODEL_PATH)

    print("\n=== Training BiggerNet ===")
    big = train_model(big, train_loader, test_loader, device, epochs=EPOCHS, lr=LR)
    big_acc = evaluate(big, test_loader, device)
    print(f"BiggerNet final test accuracy: {big_acc:.4f}")
    torch.save(big.state_dict(), BIG_MODEL_PATH)

    print("\nSaved:")
    print(f"- {SMALL_MODEL_PATH.name}")
    print(f"- {BIG_MODEL_PATH.name}")


if __name__ == "__main__":
    main()