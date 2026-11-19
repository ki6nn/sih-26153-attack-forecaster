from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from src.contracts import (
    FORECAST_STEPS,
    MITRE_STAGE_MAP,
    NUM_FEATURES,
    WINDOW_SIZE,
    ForecastOutput,
)


class AttackForecasterNet(nn.Module):
    def __init__(self, hidden_size: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(
            input_size=NUM_FEATURES,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.future_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, FORECAST_STEPS),
            nn.Sigmoid(),
        )
        self.stage_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, len(MITRE_STAGE_MAP)),
        )

    def forward(
        self, x: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        if x.shape[1:] != (WINDOW_SIZE, NUM_FEATURES):
            raise ValueError(
                f"Expected input shape (batch, {WINDOW_SIZE}, {NUM_FEATURES}), "
                f"got {tuple(x.shape)}"
            )

        sequence_output, _ = self.gru(x)
        attention_scores = self.attention(sequence_output).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), sequence_output).squeeze(1)

        future_attack_probs = self.future_head(context)
        next_stage_logits = self.stage_head(context)
        return future_attack_probs, next_stage_logits, attention_weights


def train_world_model(
    X_train,
    y_probs,
    y_stage,
    epochs: int = 15,
    batch_size: int = 64,
    save_path: str = "artifacts/world_model.pt",
) -> AttackForecasterNet:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_array = np.asarray(X_train, dtype=np.float32)
    probs_array = np.asarray(y_probs, dtype=np.float32)

    stage_names = list(MITRE_STAGE_MAP.keys())
    stage_array = np.asarray(y_stage)
    if stage_array.ndim != 1:
        stage_array = stage_array.reshape(-1)
    if np.issubdtype(stage_array.dtype, np.number):
        stage_indices = stage_array.astype(np.int64)
    else:
        stage_lookup = {name.lower(): index for index, name in enumerate(stage_names)}
        stage_indices = np.asarray(
            [stage_lookup[str(stage).lower()] for stage in stage_array],
            dtype=np.int64,
        )

    if X_array.shape[1:] != (WINDOW_SIZE, NUM_FEATURES):
        raise ValueError(
            f"X_train must have shape (N, {WINDOW_SIZE}, {NUM_FEATURES}), "
            f"got {X_array.shape}"
        )
    if probs_array.shape != (X_array.shape[0], FORECAST_STEPS):
        raise ValueError(
            f"y_probs must have shape (N, {FORECAST_STEPS}), got {probs_array.shape}"
        )
    if stage_indices.shape[0] != X_array.shape[0]:
        raise ValueError("X_train and y_stage must contain the same number of samples")
    if np.any(stage_indices < 0) or np.any(stage_indices >= len(stage_names)):
        raise ValueError(
            f"y_stage indices must be between 0 and {len(stage_names) - 1}"
        )

    dataset = TensorDataset(
        torch.from_numpy(X_array),
        torch.from_numpy(probs_array),
        torch.from_numpy(stage_indices),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AttackForecasterNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    probability_loss = nn.BCELoss()
    stage_loss = nn.CrossEntropyLoss()

    print(f"Training on {device}")
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for X_batch, probs_batch, stage_batch in loader:
            X_batch = X_batch.to(device)
            probs_batch = probs_batch.to(device)
            stage_batch = stage_batch.to(device)

            optimizer.zero_grad()
            future_probs, stage_logits, _ = model(X_batch)
            loss = (
                probability_loss(future_probs, probs_batch)
                + stage_loss(stage_logits, stage_batch)
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)

        average_loss = total_loss / len(dataset)
        print(f"Epoch {epoch:>3}/{epochs}: loss={average_loss:.6f}")

    destination = Path(save_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "window_size": WINDOW_SIZE,
            "forecast_steps": FORECAST_STEPS,
            "num_features": NUM_FEATURES,
            "stage_names": stage_names,
        },
        destination,
    )
    print(f"Saved model to {destination}")
    model.cpu()
    return model


def predict_window(
    model: AttackForecasterNet,
    window_array,
    feature_names: Sequence[str],
) -> ForecastOutput:
    if len(feature_names) != NUM_FEATURES:
        raise ValueError(
            f"feature_names must contain {NUM_FEATURES} names, "
            f"got {len(feature_names)}"
        )

    window = np.asarray(window_array, dtype=np.float32)
    if window.shape != (WINDOW_SIZE, NUM_FEATURES):
        raise ValueError(
            f"window_array must have shape ({WINDOW_SIZE}, {NUM_FEATURES}), "
            f"got {window.shape}"
        )

    device = next(model.parameters()).device
    model.eval()
    window_tensor = torch.from_numpy(window).unsqueeze(0).to(device)
    window_tensor.requires_grad_(True)

    future_probs, stage_logits, _ = model(window_tensor)
    predicted_stage_index = int(stage_logits.argmax(dim=1).item())
    predicted_next_stage = list(MITRE_STAGE_MAP.keys())[predicted_stage_index]

    attribution_score = future_probs.sum() + stage_logits[0, predicted_stage_index]
    model.zero_grad()
    attribution_score.backward()

    input_gradient = window_tensor.grad.detach().squeeze(0)
    input_values = window_tensor.detach().squeeze(0)
    feature_importance = (input_gradient * input_values).abs().sum(dim=0)
    values = feature_importance.cpu().numpy()

    ranked_indices = np.argsort(-values)
    top_indices = ranked_indices[:4]
    top_values = values[top_indices]
    max_value = float(top_values.max()) if len(top_values) else 0.0
    if max_value > 0:
        normalized_values = top_values / top_values.sum()
    else:
        normalized_values = np.zeros_like(top_values)

    top_features = [
        {
            "feature": str(feature_names[index]),
            "importance": float(normalized_values[position]),
        }
        for position, index in enumerate(top_indices)
    ]

    model.zero_grad()
    return ForecastOutput(
        current_stage="Observed Window",
        forecast_probs=[
            float(value) for value in future_probs.detach().cpu().squeeze(0).tolist()
        ],
        predicted_next_stage=predicted_next_stage,
        top_features=top_features,
    )


if __name__ == "__main__":
    torch.manual_seed(42)
    model = AttackForecasterNet()
    synthetic_input = torch.randn(2, WINDOW_SIZE, NUM_FEATURES)
    future_probs, stage_logits, attention_weights = model(synthetic_input)

    print("future_attack_probs:", tuple(future_probs.shape))
    print("next_stage_logits:", tuple(stage_logits.shape))
    print("attention_weights:", tuple(attention_weights.shape))

    feature_names = [f"feature_{index}" for index in range(NUM_FEATURES)]
    forecast = predict_window(model, torch.randn(WINDOW_SIZE, NUM_FEATURES).numpy(), feature_names)
    print("ForecastOutput:", forecast.to_dict())
