"""Tests for TemporalWorldModel, MultiTaskWorldModelLoss, and related functions.

All tests use CPU-only synthetic fixtures. No official datasets are used.
"""

import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
import torch.nn as nn

from src.contracts import (
    FEATURE_DIM,
    FORECAST_STEPS,
    IGNORE_CATEGORY_INDEX,
    NUM_ATTACK_CATEGORIES,
    NUM_FEATURES,
    WINDOW_SIZE,
    ForecastOutput,
)
from src.model.world_model import (
    AttackForecasterNet,
    MultiTaskWorldModelLoss,
    TemporalWorldModel,
    predict_window,
    train_temporal_world_model,
)


def _make_model(hidden_dim: int = 32) -> TemporalWorldModel:
    return TemporalWorldModel(
        input_dim=NUM_FEATURES,
        hidden_dim=hidden_dim,
        forecast_steps=FORECAST_STEPS,
        num_categories=NUM_ATTACK_CATEGORIES,
        dropout=0.1,
    )


class TestTemporalWorldModel(unittest.TestCase):
    """Unit tests for TemporalWorldModel architecture, rollout dynamics, and loss.

    All tests run on CPU with synthetic fixtures.
    """

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.B = 8
        self.model = _make_model()

    def test_world_model_tensor_shapes(self):
        """Input (B, 10, 20) must produce state (B,4,20), risk (B,4), cat (B,4,10)."""
        x = torch.randn(self.B, WINDOW_SIZE, NUM_FEATURES)
        states, risk, cat = self.model(x, teacher_forcing_prob=0.0)

        self.assertEqual(states.shape, (self.B, FORECAST_STEPS, NUM_FEATURES))
        self.assertEqual(risk.shape, (self.B, FORECAST_STEPS))
        self.assertEqual(cat.shape, (self.B, FORECAST_STEPS, NUM_ATTACK_CATEGORIES))

    def test_autoregressive_rollout_state_dependency(self):
        """Later horizon steps must diverge after a large perturbation to step-1 state.

        We manually run the rollout, perturbing the predicted state at step k=0
        before it is fed back as input at step k=1.  Steps k=1,2,3 must diverge from
        the unperturbed rollout because the transition cell received a different input.
        """
        x = torch.randn(self.B, WINDOW_SIZE, NUM_FEATURES)

        self.model.eval()
        with torch.no_grad():
            # Reference rollout (teacher_forcing_prob=0.0, fully autoregressive)
            orig_states, _, _ = self.model(x, teacher_forcing_prob=0.0)

            # Manual rollout with intervention after step 0
            _, h_n = self.model.encoder(x)
            h_current = h_n[-1]            # (B, hidden_dim)
            x_current = x[:, -1, :]        # (B, D)  — initial input

            states_intervened = []
            for step in range(FORECAST_STEPS):
                h_current = self.model.transition_cell(x_current, h_current)
                x_pred = self.model.state_decoder(h_current)
                states_intervened.append(x_pred)

                if step == 0:
                    # Perturb: next input is far from the predicted state.
                    # Use a very large offset (1e4) so that even after the GRUCell's
                    # sigmoid/tanh gates compress the input, the hidden state diverges.
                    x_current = x_pred + 1e4
                else:
                    x_current = x_pred

            states_intervened = torch.stack(states_intervened, dim=1)

        # Step 0 (before perturbation is applied) must be identical to reference.
        np.testing.assert_allclose(
            orig_states[:, 0, :].numpy(),
            states_intervened[:, 0, :].numpy(),
            rtol=1e-5, atol=1e-5,
        )

        # Steps 1-3 must diverge (perturbation propagates through the autoregressive chain).
        # Threshold is intentionally modest (0.01) because GRU gates may saturate.
        for k in range(1, FORECAST_STEPS):
            diff = torch.abs(orig_states[:, k, :] - states_intervened[:, k, :]).mean().item()
            self.assertGreater(
                diff, 0.01,
                msg=f"Horizon {k+1} did not diverge after step-0 perturbation (diff={diff:.6f})",
            )

    def test_teacher_forcing_vs_autoregressive(self):
        """Teacher-forced rollout (prob=1.0) must differ from autoregressive (prob=0.0)
        when target_states != the model's own predicted states.

        We use a deliberately large, random target_states tensor and call in
        training mode so scheduled sampling applies.
        """
        x = torch.randn(self.B, WINDOW_SIZE, NUM_FEATURES)
        # target_states far from model predictions → TF and AR inputs diverge
        target_states = torch.randn(self.B, FORECAST_STEPS, NUM_FEATURES) * 20.0

        self.model.train()
        with torch.no_grad():
            out_tf, _, _ = self.model(
                x, target_states=target_states, teacher_forcing_prob=1.0
            )
            out_ar, _, _ = self.model(
                x, target_states=target_states, teacher_forcing_prob=0.0
            )

        # TF and AR outputs must diverge at step > 0
        diff = torch.abs(out_tf[:, 1:, :] - out_ar[:, 1:, :]).mean().item()
        self.assertGreater(
            diff, 1e-4,
            msg="Teacher-forced and unforced rollouts should diverge",
        )

    def test_loss_masking_unsupported_stages(self):
        """IGNORE_CATEGORY_INDEX (-100) labels must produce cat_loss == 0.0 and
        zero gradient on the category decoder parameters.
        """
        loss_fn = MultiTaskWorldModelLoss()

        pred_states = torch.randn(self.B, FORECAST_STEPS, NUM_FEATURES, requires_grad=True)
        pred_risk   = torch.randn(self.B, FORECAST_STEPS, requires_grad=True)
        pred_cat    = torch.randn(self.B, FORECAST_STEPS, NUM_ATTACK_CATEGORIES, requires_grad=True)

        target_states = torch.randn(self.B, FORECAST_STEPS, NUM_FEATURES)
        target_risk   = torch.randint(0, 2, (self.B, FORECAST_STEPS)).float()
        target_cat    = torch.full(
            (self.B, FORECAST_STEPS), IGNORE_CATEGORY_INDEX, dtype=torch.long
        )

        loss, breakdown = loss_fn(
            pred_states, target_states,
            pred_risk,   target_risk,
            pred_cat,    target_cat,
        )

        self.assertEqual(breakdown["cat_loss"], 0.0, "Category loss must be zero when all labels masked")
        self.assertFalse(torch.isnan(loss).item(), "Total loss must not be NaN")

        loss.backward()
        # When all category labels are -100, CrossEntropyLoss computes no gradient
        # through pred_cat: grad is either None or all-zero.
        cat_grad = pred_cat.grad
        if cat_grad is not None:
            self.assertTrue(
                bool((cat_grad == 0.0).all().item()),
                "pred_cat.grad must be all zeros when labels are fully masked",
            )
        # (grad=None is also acceptable: no gradient was propagated to pred_cat)

    def test_loss_numerical_stability(self):
        """Clamped log-variances must keep total loss finite under extreme initialisation."""
        loss_fn = MultiTaskWorldModelLoss()

        with torch.no_grad():
            loss_fn.log_var_state.data.fill_(100.0)
            loss_fn.log_var_risk.data.fill_(-100.0)
            loss_fn.log_var_category.data.fill_(50.0)

        pred_states   = torch.randn(self.B, FORECAST_STEPS, NUM_FEATURES)
        pred_risk     = torch.randn(self.B, FORECAST_STEPS)
        pred_cat      = torch.randn(self.B, FORECAST_STEPS, NUM_ATTACK_CATEGORIES)
        target_states = torch.randn(self.B, FORECAST_STEPS, NUM_FEATURES)
        target_risk   = torch.randint(0, 2, (self.B, FORECAST_STEPS)).float()
        target_cat    = torch.randint(0, NUM_ATTACK_CATEGORIES, (self.B, FORECAST_STEPS))

        loss, _ = loss_fn(
            pred_states, target_states,
            pred_risk,   target_risk,
            pred_cat,    target_cat,
        )

        self.assertFalse(torch.isnan(loss).item())
        self.assertFalse(torch.isinf(loss).item())
        self.assertGreater(loss.item(), 0.0)

    def test_world_model_save_and_load(self):
        """State-dict round-trip must reproduce identical forward pass outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "world_model.pt"
            x = torch.randn(self.B, WINDOW_SIZE, NUM_FEATURES)

            self.model.eval()
            with torch.no_grad():
                orig_s, orig_r, orig_c = self.model(x, teacher_forcing_prob=0.0)

            torch.save(self.model.state_dict(), save_path)
            self.assertTrue(save_path.exists())

            loaded = _make_model()
            loaded.load_state_dict(torch.load(save_path, weights_only=True))
            loaded.eval()

            with torch.no_grad():
                new_s, new_r, new_c = loaded(x, teacher_forcing_prob=0.0)

            np.testing.assert_allclose(orig_s.numpy(), new_s.numpy(), atol=1e-6)
            np.testing.assert_allclose(orig_r.numpy(), new_r.numpy(), atol=1e-6)
            np.testing.assert_allclose(orig_c.numpy(), new_c.numpy(), atol=1e-6)

    def test_world_model_cpu_execution(self):
        """Model must execute and emit CPU tensors without CUDA."""
        x = torch.randn(self.B, WINDOW_SIZE, NUM_FEATURES)
        states, risk, cat = self.model(x)
        self.assertEqual(states.device.type, "cpu")

    def test_train_temporal_world_model_smoke(self):
        """Training loop must complete 2 epochs and save a checkpoint."""
        n_tr, n_vl = 32, 16

        X_tr  = np.random.randn(n_tr, WINDOW_SIZE, NUM_FEATURES).astype(np.float32)
        y_tr  = np.random.randint(0, 2, (n_tr, FORECAST_STEPS)).astype(np.float32)
        Xt_tr = np.random.randn(n_tr, FORECAST_STEPS, NUM_FEATURES).astype(np.float32)
        yc_tr = np.random.randint(0, NUM_ATTACK_CATEGORIES, (n_tr, FORECAST_STEPS)).astype(np.int64)

        X_vl  = np.random.randn(n_vl, WINDOW_SIZE, NUM_FEATURES).astype(np.float32)
        y_vl  = np.random.randint(0, 2, (n_vl, FORECAST_STEPS)).astype(np.float32)
        Xt_vl = np.random.randn(n_vl, FORECAST_STEPS, NUM_FEATURES).astype(np.float32)
        yc_vl = np.random.randint(0, NUM_ATTACK_CATEGORIES, (n_vl, FORECAST_STEPS)).astype(np.int64)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "smoke.pt"
            model, history_dict = train_temporal_world_model(
                X_train=X_tr,
                X_target_train=Xt_tr,
                y_risk_train=y_tr,
                y_cat_train=yc_tr,
                X_val=X_vl,
                X_target_val=Xt_vl,
                y_risk_val=y_vl,
                y_cat_val=yc_vl,
                epochs=2,
                batch_size=16,
                save_path=save_path,
                device="cpu",
            )

            self.assertIn("history", history_dict)
            self.assertEqual(len(history_dict["history"]), 2)
            self.assertTrue(save_path.exists())

        # Model must produce valid outputs after training
        x = torch.randn(4, WINDOW_SIZE, NUM_FEATURES)
        model.eval()
        with torch.no_grad():
            s, r, c = model(x)
        self.assertEqual(s.shape, (4, FORECAST_STEPS, NUM_FEATURES))


class TestLegacyAdapters(unittest.TestCase):
    """Backward-compatibility adapters must maintain prior contracts."""

    def test_legacy_attack_forecaster_net_shapes(self):
        """AttackForecasterNet.forward must return (probs, stage_logits, attention)."""
        model = AttackForecasterNet(hidden_size=16)
        x = torch.randn(4, WINDOW_SIZE, NUM_FEATURES)
        probs, stage_logits, attn = model(x)
        self.assertEqual(probs.shape, (4, FORECAST_STEPS))
        self.assertEqual(stage_logits.shape, (4, NUM_ATTACK_CATEGORIES))
        self.assertEqual(attn.shape, (4, WINDOW_SIZE))

    def test_predict_window_returns_forecast_output(self):
        """predict_window must return a ForecastOutput dataclass with correct fields."""
        model = _make_model()
        window = np.random.randn(WINDOW_SIZE, NUM_FEATURES).astype(np.float32)
        feature_names = [f"feat_{i}" for i in range(NUM_FEATURES)]
        result = predict_window(model, window, feature_names)

        self.assertIsInstance(result, ForecastOutput)
        self.assertEqual(len(result.forecast_probs), FORECAST_STEPS)
        self.assertIn(result.predicted_next_stage, [
            "Normal", "Reconnaissance", "Analysis", "Fuzzers", "Exploits",
            "DoS", "Generic", "Shellcode", "Backdoor", "Worms"
        ])
        self.assertIsInstance(result.top_features, list)
        self.assertGreater(len(result.top_features), 0)


if __name__ == "__main__":
    unittest.main()
