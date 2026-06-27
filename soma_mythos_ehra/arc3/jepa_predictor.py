"""JEPA Structure Predictor — predicts template probabilities from grid pairs.

Trained on ARC puzzle data to guide the beam search by predicting
which templates are most likely to solve a given puzzle.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class StructurePredictor(nn.Module):
    """Predicts template probabilities from structural features."""

    def __init__(self, feature_dim: int = 32, hidden_dim: int = 64, num_templates: int = 104) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_templates),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict template probabilities.

        Args:
            x: (batch, feature_dim) structural features
        Returns:
            (batch, num_templates) probabilities
        """
        return torch.sigmoid(self.net(x))


class JEPAStructureRouter:
    """Trains and uses the structure predictor for beam search guidance."""

    def __init__(self, feature_dim: int = 32, num_templates: int = 104) -> None:
        self.model = StructurePredictor(feature_dim, 64, num_templates)
        self.trained = False

    def train(self, X: torch.Tensor, Y: torch.Tensor, epochs: int = 100, lr: float = 0.001) -> dict:
        """Train the predictor on generated dataset."""
        dataset = TensorDataset(X, Y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / len(loader)
                print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

        self.trained = True
        return {"final_loss": total_loss / len(loader)}

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Predict template probabilities from features."""
        self.model.eval()
        return self.model(features)

    @torch.no_grad()
    def predict_top_k(self, features: torch.Tensor, k: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict top-K template indices and probabilities."""
        probs = self.predict(features)
        top_probs, top_indices = torch.topk(probs, min(k, probs.shape[-1]))
        return top_indices, top_probs

    def save(self, path: str) -> None:
        """Save model weights."""
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        """Load model weights."""
        self.model.load_state_dict(torch.load(path))
        self.trained = True
