"""Autoregressive Temporal World Model for defensive network state forecasting.

Architectural Guarantees:
    - Encodes chronological network state windows into latent representations.
    - Transitions hidden state recurrently conditioned on prior predicted state representations.
    - Emits decoupled outputs at each rollout step k in 1..K:
        1. Predicted next standardized network state (x_hat_k in R^D)
        2. Infiltration attack risk logit (p_hat_k in [0, 1])
        3. Attack category group logits (s_hat_k in R^10)
    - Supports scheduled sampling during training, and strict autoregressive rollout during evaluation.
    - Masks unannotated or missing category labels via IGNORE_CATEGORY_INDEX (-100).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from src.contracts import (
    ATTACK_CATEGORY_GROUPS,
    FORECAST_STEPS,
    IGNORE_CATEGORY_INDEX,
    NUM_ATTACK_CATEGORIES,
    NUM_FEATURES,
    WINDOW_SIZE,
    ForecastOutput,
)


class MultiTaskWorldModelLoss(nn.Module):
    """Numerically stable multi-task loss with learned homoscedastic uncertainty weighting.

    Loss Formulation (Kendall et al., 2018):
        L_total = 0.5 * exp(-s_state) * L_state + 0.5 * s_state
                + exp(-s_risk) * L_risk + 0.5 * s_risk
                + [exp(-s_cat) * L_cat + 0.5 * s_cat (if categorical labels present)]

    Log-variances s = log(sigma^2) are clamped to [-4.0, 4.0] to guarantee numerical stability.
    Category targets matching IGNORE_CATEGORY_INDEX (-100) are masked from loss and gradients.
    """

    def __init__(self) -> None:
        super().__init__()
        # Initialize log-variances at 0.0 (initial sigma = 1.0)
        self.log_var_state = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.log_var_risk = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.log_var_category = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        self.mse_loss = nn.MSELoss(reduction="mean")
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")
        self.ce_loss = nn.CrossEntropyLoss(reduction="mean", ignore_index=IGNORE_CATEGORY_INDEX)

    def forward(
        self,
        predicted_states: Tensor,
        target_states: Tensor,
        predicted_risk_logits: Tensor,
        target_risks: Tensor,
        predicted_category_logits: Tensor,
        target_categories: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute uncertainty-weighted multi-task loss with masking.

        Args:
            predicted_states: (B, K, D) float32 tensor
            target_states: (B, K, D) float32 tensor
            predicted_risk_logits: (B, K) float32 tensor
            target_risks: (B, K) float32 tensor
            predicted_category_logits: (B, K, 10) float32 tensor
            target_categories: (B, K) int64 tensor (with -100 for masked labels)

        Returns:
            Tuple of (total_loss_tensor, loss_components_dict)
        """
        # Clamp log variances for numerical safety
        s_state = torch.clamp(self.log_var_state, -4.0, 4.0)
        s_risk = torch.clamp(self.log_var_risk, -4.0, 4.0)
        s_cat = torch.clamp(self.log_var_category, -4.0, 4.0)

        # 1. State reconstruction loss (continuous flow feature dynamics)
        loss_state = self.mse_loss(predicted_states, target_states)

        # 2. Infiltration risk loss (binary attack probability)
        loss_risk = self.bce_loss(predicted_risk_logits, target_risks)

        # 3. Categorical stage loss (masked for unobserved labels)
        flat_cat_logits = predicted_category_logits.view(-1, NUM_ATTACK_CATEGORIES)
        flat_cat_targets = target_categories.view(-1)
        valid_cat_mask = flat_cat_targets != IGNORE_CATEGORY_INDEX

        if valid_cat_mask.any():
            loss_cat = self.ce_loss(flat_cat_logits, flat_cat_targets)
            cat_weighted = torch.exp(-s_cat) * loss_cat + 0.5 * s_cat
            cat_loss_val = float(loss_cat.item())
        else:
            loss_cat = torch.tensor(0.0, device=predicted_states.device)
            cat_weighted = torch.tensor(0.0, device=predicted_states.device)
            cat_loss_val = 0.0

        # Weighted combination
        total_loss = (
            0.5 * torch.exp(-s_state) * loss_state
            + 0.5 * s_state
            + torch.exp(-s_risk) * loss_risk
            + 0.5 * s_risk
            + cat_weighted
        )

        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_state": float(loss_state.item()),
            "loss_risk": float(loss_risk.item()),
            "loss_category": cat_loss_val,
            "cat_loss": cat_loss_val,
            "state_loss": float(loss_state.item()),
            "risk_loss": float(loss_risk.item()),
            "sigma_state": float(torch.exp(0.5 * s_state).item()),
            "sigma_risk": float(torch.exp(0.5 * s_risk).item()),
            "sigma_category": float(torch.exp(0.5 * s_cat).item()),
        }
        return total_loss, metrics


class TemporalWorldModel(nn.Module):
    """Autoregressive Recurrent World Model for network flow trajectory forecasting.

    Structure:
        - Sequence Encoder: 2-layer GRU over historical window (W=10, D=20).
        - Transition Cell: GRUCell projecting latent state h_{t+k} from prior state x_{t+k-1}.
        - State Decoder: Emits predicted standardized feature vector x_hat_{t+k} in R^20.
        - Risk Decoder: Emits continuous infiltration risk logit in R.
        - Category Decoder: Emits 10-class tactical attack category group logits.
    """

    def __init__(
        self,
        input_dim: int = NUM_FEATURES,
        hidden_dim: int = 128,
        forecast_steps: int = FORECAST_STEPS,
        num_categories: int = NUM_ATTACK_CATEGORIES,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forecast_steps = forecast_steps
        self.num_categories = num_categories

        # 1. State Sequence Encoder
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0.0 else 0.0,
        )

        # 2. Recurrent State-Transition Cell (Autoregressive Rollout Engine)
        self.transition_cell = nn.GRUCell(
            input_size=input_dim,
            hidden_size=hidden_dim,
        )

        # 3. Decoupled Emission Heads
        self.state_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, input_dim),
        )

        self.risk_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.category_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_categories),
        )

    def forward(
        self,
        x: Tensor,
        target_states: Tensor | None = None,
        teacher_forcing_prob: float = 0.0,
        steps: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Execute autoregressive multi-step rollout.

        Args:
            x: Input sequence tensor of shape (batch, W, D).
            target_states: Optional ground truth future states of shape (batch, K, D)
                for scheduled sampling during training.
            teacher_forcing_prob: Probability in [0, 1] of sampling ground-truth prior state.
                Strictly 0.0 during evaluation / inference.
            steps: Number of rollout steps K (defaults to self.forecast_steps=4).

        Returns:
            Tuple of:
                - predicted_states: (batch, K, D) float32 predicted feature states
                - risk_logits: (batch, K) float32 infiltration risk logits
                - category_logits: (batch, K, 10) float32 category group logits
        """
        k_steps = steps or self.forecast_steps
        if x.ndim != 3 or x.shape[1:] != (WINDOW_SIZE, self.input_dim):
            raise ValueError(
                f"Expected input shape (batch, {WINDOW_SIZE}, {self.input_dim}), got {tuple(x.shape)}"
            )

        batch_size = x.size(0)
        device = x.device

        # 1. Encode historical sequence into initial latent state h_t
        _, h_n = self.encoder(x)
        # Use top-layer hidden state as the initial transition state
        h_current = h_n[-1]  # Shape: (B, hidden_dim)

        # Initial transition input is the final observed flow in the input window
        x_current = x[:, -1, :]  # Shape: (B, input_dim)

        rollout_states: list[Tensor] = []
        rollout_risk_logits: list[Tensor] = []
        rollout_category_logits: list[Tensor] = []

        # 2. Autoregressive Rollout Loop (k = 1 .. K)
        for step_idx in range(k_steps):
            # Advance latent state dynamics conditioned on prior state
            h_current = self.transition_cell(x_current, h_current)

            # Emit decoupled predictions for current future step
            x_pred = self.state_decoder(h_current)
            risk_logit = self.risk_decoder(h_current).squeeze(-1)
            cat_logits = self.category_decoder(h_current)

            rollout_states.append(x_pred)
            rollout_risk_logits.append(risk_logit)
            rollout_category_logits.append(cat_logits)

            # Determine input for next transition step:
            # Scheduled sampling applies ONLY during training when target_states is supplied
            if (
                target_states is not None
                and teacher_forcing_prob > 0.0
                and step_idx < (k_steps - 1)
            ):
                use_teacher_forcing = (
                    teacher_forcing_prob >= 1.0 or torch.rand(1).item() < teacher_forcing_prob
                )
                if use_teacher_forcing:
                    x_current = target_states[:, step_idx, :]
                else:
                    x_current = x_pred
            else:
                # Fully autoregressive: next step is conditioned strictly on model-predicted state
                x_current = x_pred

        # Stack across horizon dimension (B, K, ...)
        out_states = torch.stack(rollout_states, dim=1)
        out_risks = torch.stack(rollout_risk_logits, dim=1)
        out_categories = torch.stack(rollout_category_logits, dim=1)

        return out_states, out_risks, out_categories

    def rollout(
        self,
        x: Tensor,
        steps: int = FORECAST_STEPS,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convenience evaluation rollout producing NumPy arrays with sigmoid risks."""
        self.eval()
        with torch.no_grad():
            states, risk_logits, cat_logits = self.forward(
                x, target_states=None, teacher_forcing_prob=0.0, steps=steps
            )
            risk_probs = torch.sigmoid(risk_logits)
            return (
                states.cpu().numpy(),
                risk_probs.cpu().numpy(),
                cat_logits.cpu().numpy(),
            )


# Adapter class preserving legacy AttackForecasterNet interface for existing UI/contracts
class AttackForecasterNet(TemporalWorldModel):
    """Adapter subclass ensuring full backward compatibility with legacy scripts."""

    def __init__(
        self,
        num_features: int = NUM_FEATURES,
        input_dim: int = NUM_FEATURES,
        hidden_size: int = 128,
        hidden_dim: int = 128,
        forecast_steps: int = FORECAST_STEPS,
        num_categories: int = NUM_ATTACK_CATEGORIES,
        dropout: float = 0.1,
    ) -> None:
        dim = input_dim if input_dim != NUM_FEATURES else num_features
        h_dim = hidden_dim if hidden_dim != 128 else hidden_size
        super().__init__(
            input_dim=dim,
            hidden_dim=h_dim,
            forecast_steps=forecast_steps,
            num_categories=num_categories,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Legacy forward adapter returning (future_attack_probs, next_stage_logits, dummy_attention)."""
        states, risk_logits, cat_logits = super().forward(x, teacher_forcing_prob=0.0)
        future_probs = torch.sigmoid(risk_logits)
        # Squeeze immediate step 1 stage logits for legacy callers
        stage_logits = cat_logits[:, 0, :]
        dummy_attention = torch.ones((x.size(0), WINDOW_SIZE), device=x.device) / WINDOW_SIZE
        return future_probs, stage_logits, dummy_attention


def train_temporal_world_model(
    X_train: np.ndarray,
    X_target_train: np.ndarray,
    y_risk_train: np.ndarray,
    y_cat_train: np.ndarray,
    X_val: np.ndarray,
    X_target_val: np.ndarray,
    y_risk_val: np.ndarray,
    y_cat_val: np.ndarray,
    epochs: int = 15,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    teacher_forcing_decay_epochs: int = 10,
    save_path: str | Path = "artifacts/world_model.pt",
    seed: int = 42,
    device: str | None = None,
) -> tuple[TemporalWorldModel, dict[str, Any]]:
    """Train TemporalWorldModel with scheduled sampling and validation-only early stopping."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Convert training tensors
    train_dataset = TensorDataset(
        torch.from_numpy(X_train.astype(np.float32)),
        torch.from_numpy(X_target_train.astype(np.float32)),
        torch.from_numpy(y_risk_train.astype(np.float32)),
        torch.from_numpy(y_cat_train.astype(np.int64)),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val.astype(np.float32)),
        torch.from_numpy(X_target_val.astype(np.float32)),
        torch.from_numpy(y_risk_val.astype(np.float32)),
        torch.from_numpy(y_cat_val.astype(np.int64)),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = TemporalWorldModel(hidden_dim=128).to(dev)
    loss_fn = MultiTaskWorldModelLoss().to(dev)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=learning_rate,
    )

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_model_state: dict[str, Any] | None = None

    for epoch in range(1, epochs + 1):
        # Linear scheduled sampling decay: 1.0 -> 0.0 over teacher_forcing_decay_epochs
        if teacher_forcing_decay_epochs > 0:
            tf_prob = max(0.0, 1.0 - (epoch - 1) / float(teacher_forcing_decay_epochs))
        else:
            tf_prob = 0.0

        model.train()
        train_loss_accum = 0.0
        train_batches = 0

        for x_b, x_tgt_b, y_risk_b, y_cat_b in train_loader:
            x_b = x_b.to(dev)
            x_tgt_b = x_tgt_b.to(dev)
            y_risk_b = y_risk_b.to(dev)
            y_cat_b = y_cat_b.to(dev)

            optimizer.zero_grad()
            p_states, p_risk_logits, p_cat_logits = model(
                x_b, target_states=x_tgt_b, teacher_forcing_prob=tf_prob
            )
            loss, _ = loss_fn(
                p_states, x_tgt_b, p_risk_logits, y_risk_b, p_cat_logits, y_cat_b
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss_accum += loss.item() * x_b.size(0)
            train_batches += 1

        avg_train_loss = train_loss_accum / len(train_dataset)

        # Validation (Strictly autoregressive: tf_prob = 0.0)
        model.eval()
        val_loss_accum = 0.0
        with torch.no_grad():
            for x_b, x_tgt_b, y_risk_b, y_cat_b in val_loader:
                x_b = x_b.to(dev)
                x_tgt_b = x_tgt_b.to(dev)
                y_risk_b = y_risk_b.to(dev)
                y_cat_b = y_cat_b.to(dev)

                p_states, p_risk_logits, p_cat_logits = model(
                    x_b, target_states=None, teacher_forcing_prob=0.0
                )
                v_loss, _ = loss_fn(
                    p_states, x_tgt_b, p_risk_logits, y_risk_b, p_cat_logits, y_cat_b
                )
                val_loss_accum += v_loss.item() * x_b.size(0)

        avg_val_loss = val_loss_accum / len(val_dataset)
        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "teacher_forcing_prob": tf_prob,
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    dest = Path(save_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "loss_fn_state_dict": loss_fn.state_dict(),
            "window_size": WINDOW_SIZE,
            "forecast_steps": FORECAST_STEPS,
            "num_features": NUM_FEATURES,
            "num_categories": NUM_ATTACK_CATEGORIES,
            "category_groups": ATTACK_CATEGORY_GROUPS,
            "best_val_loss": best_val_loss,
        },
        dest,
    )
    model.cpu()
    return model, {"history": history, "best_val_loss": best_val_loss, "saved_path": str(dest)}


# Legacy training wrapper for backward compatibility with existing main.py callers
def train_world_model(
    X_train,
    y_probs,
    y_stage,
    epochs: int = 15,
    batch_size: int = 64,
    save_path: str = "artifacts/world_model.pt",
) -> AttackForecasterNet:
    """Legacy wrapper adapting legacy inputs to train the modern model."""
    X_arr = np.asarray(X_train, dtype=np.float32)
    probs_arr = np.asarray(y_probs, dtype=np.float32)
    stage_arr = np.asarray(y_stage)

    # Synthetic future target states fallback if caller only provides historical X
    # Construct surrogate future target by repeating last observed state
    X_target = np.tile(X_arr[:, -1:, :], (1, FORECAST_STEPS, 1))

    if stage_arr.ndim == 1:
        stage_targets = np.tile(stage_arr[:, np.newaxis], (1, FORECAST_STEPS))
    else:
        stage_targets = stage_arr

    # Synthesize internal validation partition from 20% slice of training
    n = len(X_arr)
    split = int(n * 0.8)
    if split <= 0 or split >= n:
        split = n

    model, _ = train_temporal_world_model(
        X_train=X_arr[:split],
        X_target_train=X_target[:split],
        y_risk_train=probs_arr[:split],
        y_cat_train=stage_targets[:split],
        X_val=X_arr[split:] if split < n else X_arr[:split],
        X_target_val=X_target[split:] if split < n else X_target[:split],
        y_risk_val=probs_arr[split:] if split < n else probs_arr[:split],
        y_cat_val=stage_targets[split:] if split < n else stage_targets[:split],
        epochs=epochs,
        batch_size=batch_size,
        save_path=save_path,
    )
    # Return as legacy adapter
    net = AttackForecasterNet()
    net.load_state_dict(model.state_dict())
    return net


def predict_window(
    model: nn.Module,
    window_array,
    feature_names: Sequence[str],
) -> ForecastOutput:
    """Predict future risks and top contributing features for a single historical window."""
    window = np.asarray(window_array, dtype=np.float32)
    if window.shape != (WINDOW_SIZE, NUM_FEATURES):
        raise ValueError(
            f"window_array must have shape ({WINDOW_SIZE}, {NUM_FEATURES}), got {window.shape}"
        )

    model.eval()
    window_tensor = torch.from_numpy(window).unsqueeze(0)
    window_tensor.requires_grad_(True)

    if isinstance(model, TemporalWorldModel):
        states, risk_logits, cat_logits = model(window_tensor)
        future_probs = torch.sigmoid(risk_logits)
        predicted_stage_idx = int(cat_logits[0, 0, :].argmax().item())
        predicted_next_stage = ATTACK_CATEGORY_GROUPS[predicted_stage_idx]
        attribution_score = future_probs.sum() + cat_logits[0, 0, predicted_stage_idx]
    else:
        future_probs, stage_logits, _ = model(window_tensor)
        predicted_stage_idx = int(stage_logits.argmax(dim=1).item())
        predicted_next_stage = ATTACK_CATEGORY_GROUPS[predicted_stage_idx]
        attribution_score = future_probs.sum() + stage_logits[0, predicted_stage_idx]

    model.zero_grad()
    attribution_score.backward()

    input_gradient = window_tensor.grad.detach().squeeze(0)
    input_values = window_tensor.detach().squeeze(0)
    feature_importance = (input_gradient * input_values).abs().sum(dim=0).cpu().numpy()

    ranked_indices = np.argsort(-feature_importance)[:4]
    top_values = feature_importance[ranked_indices]
    norm_values = top_values / top_values.sum() if top_values.sum() > 0 else np.zeros_like(top_values)

    top_features = [
        {"feature": str(feature_names[idx]), "importance": float(norm_values[pos])}
        for pos, idx in enumerate(ranked_indices)
    ]
    model.zero_grad()

    return ForecastOutput(
        current_stage="Observed Window",
        forecast_probs=[float(v) for v in future_probs.detach().squeeze(0).tolist()],
        predicted_next_stage=predicted_next_stage,
        top_features=top_features,
    )
