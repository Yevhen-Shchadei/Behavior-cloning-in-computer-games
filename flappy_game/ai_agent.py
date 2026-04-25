import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Import ResidualNet
sys.path.insert(0, str(Path(__file__).parent.parent))
from yarik_topology import ResidualNet


class FlappyNet(nn.Module):
    def __init__(self, in_features=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.net(x)


class BiggerFlappyNet(nn.Module):
    def __init__(self, in_features=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)


class AIAgent:
    def __init__(self, model_path, meta_path, threshold=0.15):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.input_dim = int(meta["input_dim"])
        self.model_type = meta.get("model_type", "small")
        self.threshold = threshold

        # Check if it's a ResidualNet model (if model_type doesn't exist, try ResidualNet first)
        try:
            self.model = ResidualNet(
                input_dim=self.input_dim, hidden=32, num_classes=2
            ).to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded ResidualNet model with input_dim={self.input_dim}")
        except Exception as e:
            # Fallback to original models
            if self.model_type == "bigger":
                self.model = BiggerFlappyNet(in_features=self.input_dim).to(self.device)
            else:
                self.model = FlappyNet(in_features=self.input_dim).to(self.device)

            try:
                self.model.load_state_dict(
                    torch.load(model_path, map_location=self.device)
                )
                print(f"Loaded {self.model_type} FlappyNet model")
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load model: {e}\nFallback also failed: {e2}"
                )

        self.model.eval()

    def predict_action(self, features):
        x = np.asarray(features, dtype=np.float32).reshape(-1)

        if x.shape[0] != self.input_dim:
            raise ValueError(
                f"Wrong number of features. Expected {self.input_dim}, got {x.shape[0]}"
            )

        x = torch.tensor(x.reshape(1, -1), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]
            flap_prob = float(probs[1].item())

        action = 1 if flap_prob >= self.threshold else 0
        return action, flap_prob
