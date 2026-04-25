import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from yarik_topology import ResidualNet


DATA_PATH = "flappy_game/data/flappy_dataset.csv"
MODEL_PATH = "flappy_game/data/flappy_model_Y.pt"
META_PATH = "flappy_game/data/flappy_model_meta_Y.json"

SEED = 42
BATCH_SIZE = 256
EPOCHS = 35
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(path):
    df = pd.read_csv(path).dropna().copy()

    required = [
        "bird_y_norm",
        "bird_vel_norm",
        "dist_to_pipe_x_norm",
        "pipe_top_y_norm",
        "pipe_bottom_y_norm",
        "pipe_gap_center_y_norm",
        "action",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df = df[df["action"].isin([0, 1])].copy()

    # derive 9 better features
    bird_center_y = df["bird_y_norm"].astype(np.float32)
    bird_vel = df["bird_vel_norm"].astype(np.float32)
    dist_x = df["dist_to_pipe_x_norm"].astype(np.float32)
    pipe_top_y = df["pipe_top_y_norm"].astype(np.float32)
    pipe_bottom_y = df["pipe_bottom_y_norm"].astype(np.float32)
    pipe_gap_center = df["pipe_gap_center_y_norm"].astype(np.float32)

    df_features = pd.DataFrame({
        "bird_center_y": bird_center_y,
        "bird_vel": bird_vel,
        "dist_x": dist_x,
        "pipe_top_y": pipe_top_y,
        "pipe_bottom_y": pipe_bottom_y,
        "pipe_gap_center": pipe_gap_center,
        "bird_to_gap_center": bird_center_y - pipe_gap_center,
        "bird_to_top": bird_center_y - pipe_top_y,
        "bird_to_bottom": pipe_bottom_y - bird_center_y,
        "action": df["action"].astype(np.int64),
    })

    # balance dataset
    count_0 = int((df_features["action"] == 0).sum())
    count_1 = int((df_features["action"] == 1).sum())

    print(f"Before balancing: 0={count_0}, 1={count_1}")

    if count_1 == 0:
        raise ValueError("Dataset has no flap actions (action=1).")

    if count_0 > count_1 * 3:
        df_0 = df_features[df_features["action"] == 0].sample(
            n=count_1 * 3,
            random_state=SEED
        )
        df_1 = df_features[df_features["action"] == 1]
        df_features = pd.concat([df_0, df_1], axis=0).sample(
            frac=1.0,
            random_state=SEED
        ).reset_index(drop=True)

    count_0_after = int((df_features["action"] == 0).sum())
    count_1_after = int((df_features["action"] == 1).sum())
    print(f"After balancing:  0={count_0_after}, 1={count_1_after}")

    feature_cols = [
        "bird_center_y",
        "bird_vel",
        "dist_x",
        "pipe_top_y",
        "pipe_bottom_y",
        "pipe_gap_center",
        "bird_to_gap_center",
        "bird_to_top",
        "bird_to_bottom",
    ]

    X = df_features[feature_cols].astype(np.float32).values
    y = df_features["action"].astype(np.int64).values

    return X, y, feature_cols


def build_loaders(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )

    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, X_train, X_val, y_train, y_val


def build_model(input_dim):
    return ResidualNet(input_dim=input_dim, hidden=32, num_classes=2).to(DEVICE)


def train():
    set_seed(SEED)

    X, y, feature_cols = load_dataset(DATA_PATH)
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Feature cols: {feature_cols}")

    train_loader, val_loader, X_train, X_val, y_train, y_val = build_loaders(X, y)

    model = build_model(X.shape[1])

    class_counts = np.bincount(y_train)
    class_weights = class_counts.sum() / (len(class_counts) * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)

            train_preds.extend(preds.detach().cpu().numpy())
            train_targets.extend(yb.detach().cpu().numpy())

        train_loss /= len(train_loader.dataset)
        train_acc = accuracy_score(train_targets, train_preds)

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)

                logits = model(xb)
                loss = criterion(logits, yb)

                val_loss += loss.item() * xb.size(0)
                preds = torch.argmax(logits, dim=1)

                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(yb.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(val_targets, val_preds)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    print(f"\nBest val_acc: {best_val_acc:.4f}")

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(X_val, dtype=torch.float32, device=DEVICE))
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    print("\nValidation report:")
    print(classification_report(y_val, preds, digits=4))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(best_state, MODEL_PATH)

    meta = {
        "input_dim": int(X.shape[1]),
        "feature_cols": feature_cols,
        "feature_mode": "derived_9_features",
        "device": DEVICE,
    }

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved meta: {META_PATH}")


if __name__ == "__main__":
    train()