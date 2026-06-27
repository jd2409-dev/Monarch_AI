"""Learned Template Classifier — trains on ARC data to predict winning templates.

Analyzes input/output grid pairs and predicts which template programs
are most likely to solve the puzzle, enabling aggressive search pruning.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn

from soma_mythos_ehra.arc3.dsl_kernel import DSLKernel, DSLNode
from soma_mythos_ehra.arc3.template_library import build_template_library


@dataclass
class ClassifierConfig:
    feature_dim: int = 64
    hidden_dim: int = 128
    num_templates: int = 100
    learning_rate: float = 0.001
    epochs: int = 50


class GridFeatureExtractor(nn.Module):
    """Extracts fixed-size features from grid pairs."""

    def __init__(self, feature_dim: int = 64) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(8),
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, feature_dim),
            nn.ReLU(),
        )

    def forward(self, inp: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """Extract features from input/output pair.

        Args:
            inp: (H, W) input grid
            out: (H, W) output grid
        Returns:
            (feature_dim,) feature vector
        """
        # Normalize to [0, 1]
        inp_norm = inp.float() / 9.0
        out_norm = out.float() / 9.0

        # Pad to same size
        max_h = max(inp.shape[0], out.shape[0])
        max_w = max(inp.shape[1], out.shape[1])

        inp_padded = torch.zeros(max_h, max_w)
        out_padded = torch.zeros(max_h, max_w)
        inp_padded[:inp.shape[0], :inp.shape[1]] = inp_norm
        out_padded[:out.shape[0], :out.shape[1]] = out_norm

        # Stack as channels: (2, H, W)
        stacked = torch.stack([inp_padded, out_padded], dim=0).unsqueeze(0)
        return self.conv(stacked).squeeze(0)


class TemplateClassifier(nn.Module):
    """Predicts template probabilities from grid pairs."""

    def __init__(self, config: ClassifierConfig | None = None) -> None:
        super().__init__()
        self.config = config or ClassifierConfig()
        self.extractor = GridFeatureExtractor(self.config.feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(self.config.feature_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.num_templates),
            nn.Sigmoid(),
        )

    def forward(self, inp: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """Predict template probabilities."""
        features = self.extractor(inp, out)
        return self.classifier(features)


class LearnedClassifier:
    """Trains and uses a template classifier on ARC training data."""

    def __init__(self, config: ClassifierConfig | None = None) -> None:
        self.config = config or ClassifierConfig()
        self.templates = build_template_library()
        self.kernel = DSLKernel(background=0)
        self.model = TemplateClassifier(self.config)
        self.trained = False

    def train_on_dataset(self, data_dir: str | Path, limit: int = 200) -> dict:
        """Train the classifier on ARC training puzzles.

        For each puzzle, executes all templates and records which ones solve it.
        """
        data_dir = Path(data_dir)
        json_files = sorted(data_dir.glob("*.json"))[:limit]

        print(f"Training on {len(json_files)} puzzles with {len(self.templates)} templates")

        # Collect training data
        features_list = []
        labels_list = []

        for i, json_file in enumerate(json_files):
            if (i + 1) % 50 == 0:
                print(f"  Processing {i+1}/{len(json_files)}...")

            try:
                with open(json_file) as f:
                    data = json.load(f)

                train_pairs = data.get("train", [])
                if not train_pairs:
                    continue

                # Get first train pair
                pair = train_pairs[0]
                inp = torch.tensor(pair["input"], dtype=torch.long)
                out = torch.tensor(pair["output"], dtype=torch.long)

                # Extract features
                features = self.model.extractor(inp, out)

                # Test all templates
                label = torch.zeros(len(self.templates))
                for j, (name, prog) in enumerate(self.templates):
                    correct, _ = self.kernel.execute_on_pairs(prog, [inp], [out])
                    if correct == 1:
                        label[j] = 1.0

                # Only add if at least one template solves it
                if label.sum() > 0:
                    features_list.append(features)
                    labels_list.append(label)

            except Exception:
                continue

        if not features_list:
            print("No training data collected")
            return {"accuracy": 0.0}

        # Stack tensors
        X = torch.stack(features_list)
        Y = torch.stack(labels_list)

        print(f"Collected {len(X)} training samples")

        # Train
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.BCELoss()

        self.model.train()
        for epoch in range(self.config.epochs):
            optimizer.zero_grad()
            pred = self.model.classifier(X)
            loss = criterion(pred, Y)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                accuracy = ((pred > 0.5).float() == Y).float().mean().item()
                print(f"  Epoch {epoch+1}/{self.config.epochs}: loss={loss.item():.4f} acc={accuracy:.2%}")

        self.trained = True
        return {"accuracy": accuracy, "loss": loss.item()}

    @torch.no_grad()
    def predict_templates(self, inp: torch.Tensor, out: torch.Tensor, top_k: int = 10) -> list[tuple[str, DSLNode, float]]:
        """Predict which templates are most likely to solve this puzzle.

        Returns list of (name, program, probability) sorted by probability.
        """
        self.model.eval()
        probs = self.model(inp, out)

        # Get top-K indices
        top_indices = torch.argsort(probs, descending=True)[:top_k]

        results = []
        for idx in top_indices:
            i = idx.item()
            if i < len(self.templates):
                name, prog = self.templates[i]
                prob = probs[i].item()
                results.append((name, prog, prob))

        return results


def train_classifier(data_dir: str = "ARC-AGI/data/training", limit: int = 200) -> LearnedClassifier:
    """Train the classifier and return it."""
    classifier = LearnedClassifier(ClassifierConfig(epochs=30))
    classifier.train_on_dataset(data_dir, limit)
    return classifier
