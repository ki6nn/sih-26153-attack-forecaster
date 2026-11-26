# NetworkSentinel AI: Initial Foundation Audit & Architectural Repair Roadmap

**Smart India Hackathon 2026 — Problem Statement 26153**  
**Project:** NetworkSentinel AI (Defensive Network Traffic Analysis & Forecasting)  
**Repository:** `ki6nn/sih-26153-attack-forecaster`  
**Audit Date:** 2026-11-26  
**Auditor:** Senior Python ML & Defensive Security Engineer  
**Status:** Audit & Foundation Design Complete (Zero Model Logic Rewrites Conducted)

---

## Executive Summary

NetworkSentinel AI is a defensive cybersecurity research prototype intended to perform multi-step attack forecasting and MITRE ATT&CK stage classification on raw/flow-level network traffic datasets.

This comprehensive audit inspected every module, data artifact, configuration file, and dependency across the repository. The inspection revealed that while the conceptual deep temporal architecture (`AttackForecasterNet` combining a 2-layer GRU with additive attention, multi-step future forecasting, and stage classification) and the scikit-learn baseline (`BaselineAttackClassifier`) have an established foundation, the repository suffers from critical architectural gaps, unverified synthetic data fallbacks, broken serialization, potential data leakage pathways, missing pipeline modules, and environment execution constraints on Windows.

Most critically, **no dataset ingestion or preprocessing pipeline currently exists in `src/pipeline`**, and existing serialized artifacts (`artifacts/baseline_rf.pkl`) were generated from purely synthetic Gaussian noise.

This document establishes the initial repository audit, details broken implementations, formalizes a strict dataset policy for CIC-IDS2017 and UNSW-NB15, provides the foundation engineering design, and delivers a concrete repair checklist.

---

## 1. Inventory of Inspected Artifacts & Repository Layout

The current repository layout was inspected exhaustively:

```
sih-26153-attack-forecaster/
├── .gitignore                      # Configured: ignores .venv/, data/*.csv (except sample_demo.csv), artifacts/*.pt, *.pkl
├── requirements.txt                # Lists 8 core Python libraries
├── artifacts/
│   └── baseline_rf.pkl             # Serialized joblib artifact (1,160,300 bytes)
├── data/                           # Directory present; completely empty (0 files)
├── docs/                           # Created for audit and architectural documentation
│   └── initial_audit.md            # This document
└── src/
    ├── __init__.py                 # Empty package marker
    ├── contracts.py                # System constants, MITRE mapping, dataclasses, mock generator
    ├── dashboard/
    │   └── __init__.py             # Empty package marker (0 bytes; no dashboard code)
    ├── pipeline/
    │   └── __init__.py             # Empty package marker (0 bytes; no ingestion/preprocessing code)
    └── model/
        ├── __init__.py             # Empty package marker
        ├── baseline.py             # Random Forest binary & multi-class wrapper + evaluate_models()
        └── world_model.py          # PyTorch GRU + Attention multi-step forecaster + saliency
```

### Git Branch & Environment Status
- **Git HEAD:** Commit `cd89ed0` on `feat/world-model`.
- **Target Working Branch:** `development/repair`. Note: In this sandboxed Windows environment, `.git/refs/heads/` directory locking prevents branch creation directly via sandboxed git commands (`Permission denied` on `.git/refs/heads/development_repair.lock`). All repair work is isolated within workspace working files without modifying `main`.
- **Python Runtime:** Python 3.12.10 (AMD64) in `.venv`.
- **Virtual Environment Packages:** `pandas==2.2.3`, `numpy==2.5.5`, `torch==2.15.0`, `scikit-learn==1.9.1`, `scipy==1.18.1`, `shap==0.54.1`, `streamlit==1.67.0`, `plotly==7.1.0`, `joblib==1.6.0`.

---

## 2. Existing Architecture & Functionality

### 2.1 System Contracts (`src/contracts.py`)
- **Constants:**
  - `WINDOW_SIZE = 10`: Temporal lookback sequence length ($t-9 \dots t$).
  - `FORECAST_STEPS = 4`: Forecasting horizon ($t+1 \dots t+4$).
  - `NUM_FEATURES = 20`: Hardcoded expected feature vector dimension.
- **MITRE ATT&CK Mapping (`MITRE_STAGE_MAP`):**
  Maps 10 stages (`Normal`, `Reconnaissance`, `Analysis`, `Fuzzers`, `Exploits`, `DoS`, `Generic`, `Shellcode`, `Backdoor`, `Worms`) to MITRE tactics (e.g. `TA0043`, `TA0007`, `TA0002`, `TA0040`, `TA0005`, `TA0003`, `TA0008`) and technique IDs.
- **Output Schema (`ForecastOutput`):**
  Dataclass holding `current_stage`, `forecast_probs` ($H=4$), `predicted_next_stage`, and `top_features` (list of feature-importance dictionaries).
- **Mock Fallback (`get_mock_forecast()`):**
  Provides static, hardcoded predictions (`0.58, 0.24, 0.12, 0.06`, `Exploits`, and UNSW-NB15 flow features: `sbytes`, `dbytes`, `dur`, `rate`, `sload`).

### 2.2 Baseline Classifier (`src/model/baseline.py`)
- **Architecture:** Dual `RandomForestClassifier(n_estimators=200, class_weight='balanced')`:
  1. `attack_model`: Binary classification ($0 = \text{Benign/Normal}, 1 = \text{Attack}$).
  2. `stage_model`: Multiclass classification over MITRE stages.
- **Feature Adaptor (`_prepare_features`):**
  Accepts 3D tensors $(N, 10, 20)$ and slices the final temporal slice $(:, -1, :)$ to evaluate a static baseline on the latest time step. Accepts 2D $(N, 20)$ directly.
- **Label Encoder (`_encode_stage_labels`):**
  Maps case-insensitive stage strings to contiguous integer indices according to `MITRE_STAGE_MAP`.
- **Evaluation Utility (`evaluate_models`):**
  Calculates Accuracy, Precision, Recall, F1-Score, and ROC-AUC for both the World Model and Baseline RF against ground-truth binary attack labels.

### 2.3 World Model Forecaster (`src/model/world_model.py`)
- **PyTorch Architecture (`AttackForecasterNet`):**
  - Input: $(B, W=10, D=20)$.
  - Recurrent Backbone: 2-layer `nn.GRU(input_size=20, hidden_size=128, batch_first=True, dropout=0.2)`.
  - Attention Mechanism: Additive self-attention scoring sequence steps via `nn.Sequential(Linear(128, 128), Tanh(), Linear(128, 1))` followed by Softmax along time dimension, producing pooled context vector $(B, 128)$.
  - Multi-Step Forecast Head (`future_head`): `Linear(128, 64) -> ReLU() -> Linear(64, 4) -> Sigmoid()`, outputting binary probabilities across the next 4 steps.
  - Stage Head (`stage_head`): `Linear(128, 64) -> ReLU() -> Linear(64, 10)`, outputting unnormalized stage classification logits.
- **Training Loop (`train_world_model`):**
  Trains with combined objective $\mathcal{L} = \mathcal{L}_{\text{BCE}}(\hat{p}_{1..4}, y_{1..4}) + \mathcal{L}_{\text{CE}}(\hat{z}_{\text{stage}}, y_{\text{stage}})$.
- **Feature Saliency Attribution (`predict_window`):**
  Uses gradient-input attribution: computes $\nabla_{X} (\sum \hat{p} + \hat{z}_{\text{stage}})$, multiplies by input values, sums absolute values across time steps, ranks top 4 features, and normalizes them.

---

## 3. Broken, Questionable, & Fragile Implementations

### 3.1 Critical Serialization Bug in Baseline Model (`artifacts/baseline_rf.pkl`)
- **Issue:** `artifacts/baseline_rf.pkl` cannot be unpickled in standard production contexts:
  ```
  AttributeError: Can't get attribute 'BaselineAttackClassifier' on <module '__main__'>
  ```
- **Root Cause:** In `src/model/baseline.py`, the file's `__main__` block executed `baseline.fit(...)` and serialized `self` directly using `joblib.dump(self, ...)`. When a class is dumped inside `__main__`, Python records its module path as `__main__.BaselineAttackClassifier`. If another script attempts `joblib.load("artifacts/baseline_rf.pkl")`, Python searches for the class definition inside the importing caller's `__main__` namespace rather than `src.model.baseline`.
- **Synthetic Contamination:** Furthermore, binary inspection confirmed that `artifacts/baseline_rf.pkl` was fitted entirely on 120 synthetic Gaussian samples (`np.random.default_rng(42).normal(size=(120, 20))`).

### 3.2 Single-Class Probability Inversion in Baseline
- **Location:** `src/model/baseline.py`, lines 112–115:
  ```python
  probabilities = self.attack_model.predict_proba(self._prepare_features(features))
  if probabilities.shape[1] == 2:
      return probabilities[:, 1]
  return probabilities[:, 0]
  ```
- **Flaw:** If a mini-batch or dataset split contains only one class (e.g. during an all-benign window or pure baseline evaluation), `predict_proba` returns an array of shape $(N, 1)$. If that single class was class `0` (Normal), falling back to `probabilities[:, 0]` assigns the benign probability as the attack probability, reporting a $1.0$ attack probability for normal traffic!

### 3.3 Flawed Multiclass Indexing in `predict_stage`
- **Location:** `src/model/baseline.py`, lines 117–121:
  ```python
  indices = self.stage_model.predict(self._prepare_features(features)).astype(int)
  return np.asarray([self.stage_names[index] for index in indices])
  ```
- **Flaw:** `self.stage_model.fit(features, stage_labels)` fits on the observed integer labels. If the training split only observed classes `0` (`Normal`) and `4` (`Exploits`), `stage_model.classes_` contains `[0, 4]`. The predictions returned by `predict()` are `0` and `4`. If any stage index is missing from training or non-contiguous, or if remapped, indexing into `self.stage_names[index]` will either fail or silently misalign if class indices are arbitrary.

### 3.4 Uncalibrated Saliency Attribution in `predict_window`
- **Location:** `src/model/world_model.py`, line 185:
  ```python
  attribution_score = future_probs.sum() + stage_logits[0, predicted_stage_index]
  ```
- **Flaw:**
  1. `future_probs.sum()` is bounded between $[0, 4.0]$ via Sigmoid activation.
  2. `stage_logits[0, predicted_stage_index]` is an unnormalized real value from an unbounded linear layer ($\in (-\infty, +\infty)$).
  Summing these two uncalibrated quantities means the stage logits will completely dominate the gradient $\nabla_X$, making the attribution solely explain stage classification while ignoring the multi-step temporal attack probability forecast.
  3. Feature importance sums absolute values across all 10 timesteps, ignoring the temporal dimension of *when* a feature spike occurred.
  4. Ranking only top 4 features and dividing by their sum discards information about the remaining 16 features and produces an artificial percentage distribution.

### 3.5 Silent Dimensional Flattening in `_prepare_features`
- **Location:** `src/model/baseline.py`, lines 50–52:
  ```python
  else:
      array = array.reshape(array.shape[0], -1)
  ```
- **Flaw:** If an array with unexpected dimensions is passed (such as a 3D tensor of shape $(N, 5, 4)$ representing 5 timesteps of 4 features), it is silently flattened to $(N, 20)$ and accepted because $5 \times 4 = 20$. This introduces catastrophic semantic corruption without raising an error.

---

## 4. Data Leakage Risks & Metric Validity

### 4.1 Temporal Boundary Overlap in Rolling Windows
- **Risk:** Creating rolling windows before dataset splitting causes severe temporal data leakage.
- **Mechanism:** If a dataset of $T$ chronological steps is windowed with $W=10$ and $H=4$, a window at index $t_k$ encompasses time steps $[t_k - 9 \dots t_k]$ with targets $[t_{k+1} \dots t_{k+4}]$. If a random or naive split point is placed at step $S$, windows originating at $S-5$ contain historical features from the training set while predicting target values in the validation/test set.
- **Requirement:** Dataset splitting into Chronological Partitions (Train, Val, Test) must be performed **strictly before** rolling window extraction. Furthermore, a temporal buffer/gap equal to $W + H - 1 = 13$ steps must separate the partition boundaries to prevent any shared auto-regressive state.

### 4.2 In-Sample Preprocessing Contamination
- **Risk:** Fitting normalization scalers, imputation values, or target encodings across the full dataset prior to train/test partitioning.
- **Current State:** The repository currently has zero preprocessing code. If future engineers add scalers directly inside scripts without strict encapsulation, min/max bounds and standard deviations from future test sets will leak into training features.
- **Requirement:** All scalers (e.g. `RobustScaler` / `StandardScaler`) and encoders must be fitted **exclusively on the training partition**. The fitted parameters must be persisted in `artifacts/preprocessor.joblib` and applied as pure read-only transforms to validation and test data.

### 4.3 Metric Distortion via `np.max` Aggregation in `evaluate_models`
- **Location:** `src/model/baseline.py`, lines 149–150:
  ```python
  if array.shape[1] > 2:
      return np.clip(np.max(array, axis=1), 0.0, 1.0)
  ```
- **Severe Distortion Risk:**
  When evaluating `world_model_probs` (shape $N \times 4$), the function reduces the 4 future forecast probabilities by taking the maximum probability across the horizon: $\max(\hat{p}_{t+1}, \hat{p}_{t+2}, \hat{p}_{t+3}, \hat{p}_{t+4})$.
  1. If `y_true` represents the ground-truth binary label at step $t+1$, comparing it against $\max_{h=1..4} \hat{p}_{t+h}$ tests two completely different concepts: "will an attack occur at $t+1$?" versus "will an attack occur anywhere in the next 4 steps?".
  2. This artificially inflates the World Model's True Positive Rate (Recall) while depressing its False Positive Rate, creating an invalid, biased comparison against the Baseline RF (which only predicts the immediate state).
  3. Metric evaluation must either evaluate step-by-step ($h=1, 2, 3, 4$) or evaluate against a matching ground-truth target vector $y_{\text{horizon}} = \bigvee_{h=1}^{4} y_{t+h}$.

### 4.4 Fixed 0.5 Decision Threshold on Severely Imbalanced Traffic
- **Risk:** In real-world intrusion detection datasets (e.g. CIC-IDS2017 Monday/Tuesday, UNSW-NB15), attacks constitute a small fraction of total flows ($<5\%$). Applying a fixed $0.5$ classification threshold leads to deceptive evaluation metrics:
  - Accuracy becomes trivially high ($>95\%$) simply by predicting all zeros.
  - F1-Score collapses to zero if uncalibrated probabilities hover around $0.3 - 0.4$.
- **Requirement:** Precision-Recall AUC (PR-AUC), Brier Score, and validation-calibrated decision thresholds must accompany standard ROC-AUC and F1 metrics.

---

## 5. Dataset Assumptions & Discrepancies

### 5.1 UNSW-NB15 vs. CIC-IDS2017 Taxonomy Conflict
- **Discrepancy:** `MITRE_STAGE_MAP` in `src/contracts.py` is hardcoded to 10 specific categories:
  `['Normal', 'Reconnaissance', 'Analysis', 'Fuzzers', 'Exploits', 'DoS', 'Generic', 'Shellcode', 'Backdoor', 'Worms']`.
- **Conflict:**
  1. These 10 categories represent the **UNSW-NB15** attack taxonomy.
  2. In raw UNSW-NB15 files, the label is pluralized as `"Backdoors"`, whereas `contracts.py` uses singular `"Backdoor"`.
  3. **CIC-IDS2017** contains completely different attack labels: `BENIGN`, `DDoS`, `PortScan`, `Bot`, `Infiltration`, `Web Attack – Brute Force`, `Web Attack – XSS`, `Web Attack – Sql Injection`, `FTP-Patator`, `SSH-Patator`, `DoS slowloris`, `DoS Slowhttptest`, `DoS Hulk`, `DoS GoldenEye`, `Heartbleed`.
  4. The current codebase has no mechanism to ingest, validate, or map CIC-IDS2017 classes to MITRE ATT&CK stages.

### 5.2 The Hardcoded `NUM_FEATURES = 20` Assumption
- **Discrepancy:**
  - Raw UNSW-NB15 has 49 features in the comprehensive 4-CSV release and 45 features in the train/test sets.
  - CIC-IDS2017 has 83 features generated by CICFlowMeter.
  - The contracts hardcode `NUM_FEATURES = 20` without specifying:
    a) Which 20 features are selected.
    b) What feature selection criterion was applied (e.g. Mutual Information, Random Forest Gini importance, or domain-driven flow characteristics).
    c) How categorical attributes (`proto`, `service`, `state`) are encoded into numeric values.

### 5.3 Concatenated Artificial Timelines in UNSW-NB15 (Strict Rule 7 Violation Risk)
- **Caution:** In UNSW-NB15, many student prototypes mistakenly concatenate `UNSW_NB15_training-set.csv` (82,332 rows) and `UNSW_NB15_testing-set.csv` (175,341 rows) to simulate a continuous timeline. As documented in literature, these files are non-contiguous, post-stratified samples. Concatenating them creates an artificial, invalid timeline.
- **Requirement:** True temporal forecasting must either use the official raw chronologically ordered captures (`UNSW-NB15_1.csv` through `UNSW-NB15_4.csv`) sorted by flow start time (`stime`), or use official per-day captures of CIC-IDS2017.

---

## 6. Dependency, Environment, & Execution Verification

### 6.1 Windows Application Control (WDAC) / Sandboxing Restriction
- **Finding:** During verification of Python imports using `.venv\Scripts\python.exe`:
  - `import numpy`, `import pandas`, `import joblib`, `import torch` succeed without errors.
  - `import scipy.special` and `import sklearn` trigger:
    ```
    ImportError: DLL load failed while importing _special_ufuncs: An Application Control policy has blocked this file.
    ```
- **Analysis:** On this Windows host, Windows Defender Application Control (WDAC) / AppLocker policy restricts dynamic execution of compiled C-extension DLLs (specifically `_special_ufuncs.pyd` in `scipy\special`) residing within user-space virtual environments.
- **Architectural Impact:**
  - PyTorch (`torch`) runs successfully.
  - Scikit-learn commands require execution environments where WDAC policies allow Python virtual environment extension loading, or execution outside the restricted policy scope.
  - All pure-Python, NumPy, Pandas, and PyTorch modules run cleanly inside the sandbox.

### 6.2 Missing Test Harness & Validation Libraries
- **Inspection:**
  - `pytest` is **not installed** in `.venv` and **not listed** in `requirements.txt`.
  - No testing framework or test suites exist in the workspace.
  - `requirements.txt` lacks pins for data validation tools (e.g. `pydantic` or `pandera`) and linting/formatting tools.

---

## 7. Dataset Policy: Classification, Governance, & Integrity

In strict adherence to Rules 5, 6, and 7, the project adopts a formal five-tier data policy. Under no circumstances will synthetic or randomized data be used as research evidence or presented in evaluation tables.

| Tier | Data Designation | Source / Derivation | Permitted Use Cases | Strict Prohibitions |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1** | **Research Data** | Official raw releases of **CIC-IDS2017** (UNB) and **UNSW-NB15** (UNSW Canberra). | Benchmark ground-truth evaluation, final ablation studies, publication-grade metrics. | Must never be modified, artificially augmented with Gaussian noise, or partially synthesized. |
| **Tier 2** | **Demo Data** | A small, contiguous, chronological slice extracted directly from official Research Data (e.g. `data/sample_demo.csv`). | UI dashboard demonstration, smoke tests, rapid verification of pipeline execution. | Must never be generated using `np.random` or `faker`. Must preserve real network relationships. |
| **Tier 3** | **Training Data** | The first chronological split ($t_0 \dots t_{\text{train}}$) representing the earliest 70% of network flow records. | Fitting model weights, updating gradient parameters, fitting scalers and encoders. | Must never overlap with validation or test time horizons. |
| **Tier 4** | **Validation Data** | The intermediate chronological split ($t_{\text{train}} \dots t_{\text{val}}$) representing the subsequent 15% of records. | Hyperparameter optimization, early stopping, decision threshold tuning ($p_{\text{thresh}}$). | Must never be used to fit feature scalers or parameter weights. |
| **Tier 5** | **Test Data** | The final chronological split ($t_{\text{val}} \dots t_{\text{end}}$) representing the last 15% of records. | Strictly held-out out-of-time evaluation of generalized forecasting ability. | Preprocessors and models must remain completely blind to test data until final reporting. |

### Handling Missing Datasets
When raw dataset files are absent from `data/`:
- The pipeline **MUST NOT** silently fall back to synthetic data generators (`torch.randn`, `rng.normal`).
- The pipeline **MUST** raise an informative, actionable `DatasetNotFoundError` instructing the operator on how to obtain and place official CSV files into `data/`.

---

## 8. Foundation Design & Architecture Proposal

To repair the foundation without disrupting existing models, we propose adding modular components inside `src/pipeline/` and `src/contracts.py`:

```
src/
├── contracts.py              # Extended with dataset schemas, stage taxonomies & validation contracts
├── pipeline/
│   ├── __init__.py           # Package exports
│   ├── ingestion.py          # Unified loader for CIC-IDS2017 & UNSW-NB15
│   ├── validation.py         # Schema, range, NaN/Inf, and temporal consistency checks
│   ├── preprocessor.py       # Training-only fit/transform pipeline with joblib persistence
│   ├── features.py           # Top-20 deterministic feature extraction & temporal aggregation
│   └── windowing.py          # Strict chronological train/val/test splitting & rolling window creation
```

### 8.1 Ingestion & Unified Normalization (`src/pipeline/ingestion.py`)
- **Dataset Agnostic Interface:**
  - Load official CSVs in memory-efficient chunks (`pd.read_csv(..., chunksize=50000)`).
  - Strip whitespace from column headers (critical for CIC-IDS2017, which contains leading/trailing spaces such as `" Destination Port"` and `" Flow Duration"`).
  - Standardize timestamp columns to UTC datetime objects and sort chronologically.

### 8.2 Dataset Validation (`src/pipeline/validation.py`)
- **Integrity Verifications:**
  1. Check for infinite values (`np.isinf`) and replace with bounded maximums or NaN before imputation.
  2. Impute missing numeric values using training medians.
  3. Validate class label distributions and enforce unified MITRE stage mapping.
  4. Ensure monotonic or sorted chronological order across flow timestamps.

### 8.3 Top-20 Feature Contract (`src/pipeline/features.py`)
To satisfy `NUM_FEATURES = 20` deterministically across datasets:
- **UNSW-NB15 Top-20 Flow Features:**
  `dur`, `sbytes`, `dbytes`, `sttl`, `dttl`, `sloss`, `dloss`, `Sload`, `Dload`, `Spkts`, `Dpkts`, `swin`, `dwin`, `stcpb`, `dtcpb`, `smeansz`, `dmeansz`, `trans_depth`, `res_bdy_len`, `sjit`.
- **CIC-IDS2017 Top-20 Flow Features:**
  `Flow Duration`, `Total Fwd Packets`, `Total Backward Packets`, `Total Length of Fwd Packets`, `Total Length of Bwd Packets`, `Fwd Packet Length Max`, `Fwd Packet Length Min`, `Fwd Packet Length Mean`, `Bwd Packet Length Max`, `Bwd Packet Length Min`, `Bwd Packet Length Mean`, `Flow Bytes/s`, `Flow Packets/s`, `Flow IAT Mean`, `Flow IAT Std`, `Fwd IAT Total`, `Bwd IAT Total`, `FIN Flag Count`, `SYN Flag Count`, `Init_Win_bytes_forward`.

### 8.4 Training-Only Preprocessing (`src/pipeline/preprocessor.py`)
- Encapsulate `RobustScaler` (resistant to extreme network volume outliers) or `StandardScaler`.
- Implement `NetworkDataPreprocessor`:
  - `fit(X_train)`: Computes medians and quantile ranges from training data only.
  - `transform(X)`: Applies pure scaling transform.
  - `save(path)` / `load(path)`: Serializes preprocessing state using joblib.

### 8.5 Chronological Splitting & Leak-Free Windowing (`src/pipeline/windowing.py`)
- **Partitioning:**
  Given a chronologically sorted DataFrame:
  $$\text{Split indices: } N_{\text{train}} = \lfloor 0.70 \cdot N \rfloor, \quad N_{\text{val}} = \lfloor 0.85 \cdot N \rfloor$$
- **Window Extraction:**
  For partition $P \in \{\text{Train}, \text{Val}, \text{Test}\}$:
  Slide window of size $W=10$ with step size $S=1$ and future horizon $H=4$:
  $$X_i = P[i : i + W], \quad y^{\text{future}}_i = P_{\text{attack}}[i + W : i + W + H], \quad y^{\text{stage}}_i = P_{\text{stage}}[i + W]$$
  No window spans across partition boundaries.

### 8.6 Reproducibility Standard
- Create `src/utils/reproducibility.py` providing `set_seed(seed=42)`:
  - Sets `random.seed(seed)`
  - Sets `np.random.seed(seed)`
  - Sets `torch.manual_seed(seed)`
  - Enables `torch.backends.cudnn.deterministic = True`

---

## 9. Proposed Implementation Checklist

- [ ] **Phase 1: Environment & Dependency Repair**
  - [ ] Add `pytest` and `pydantic` to `requirements.txt`.
  - [ ] Document Windows WDAC / AppLocker execution workarounds for scikit-learn DLLs.
- [ ] **Phase 2: Contracts & Schema Hardening (`src/contracts.py`)**
  - [ ] Add unified MITRE taxonomy supporting both UNSW-NB15 and CIC-IDS2017.
  - [ ] Define the explicit list of Top-20 features for both datasets.
  - [ ] Deprecate `get_mock_forecast()` in favor of live validation checks.
- [ ] **Phase 3: Pipeline Foundation Implementation (`src/pipeline/`)**
  - [ ] Implement `ingestion.py` (chunked reading, header normalization, timestamp sorting).
  - [ ] Implement `validation.py` (strict schema checks, Inf/NaN remediation, label integrity).
  - [ ] Implement `preprocessor.py` (training-only fitting, persistence to `artifacts/preprocessor.joblib`).
  - [ ] Implement `features.py` (deterministic Top-20 selection and encoding).
  - [ ] Implement `windowing.py` (chronological partitioning with buffer, leak-free window extraction).
- [ ] **Phase 4: Model & Serialization Fixes**
  - [ ] Fix pickling in `src/model/baseline.py` so class unpickling references `src.model.baseline`.
  - [ ] Fix single-class probability fallback in `BaselineAttackClassifier.predict_attack_proba`.
  - [ ] Calibrate feature attribution in `src/model/world_model.py` (separate probability saliency from stage logit attribution).
  - [ ] Align multi-step evaluation metrics in `evaluate_models` to prevent horizon $\max$ distortion.
- [ ] **Phase 5: Test Suite Construction (`tests/`)**
  - [ ] `tests/test_contracts.py`: Validate dimensions, constants, and MITRE stage mappings.
  - [ ] `tests/test_pipeline.py`: Test chronological ordering, zero-leakage splits, and preprocessor isolation.
  - [ ] `tests/test_baseline.py`: Test unpickling, predict shapes, and edge-case single-class batches.
  - [ ] `tests/test_world_model.py`: Test tensor shapes, loss gradients, and determinism.
- [ ] **Phase 6: Real Dataset Ingestion & Verification**
  - [ ] Ingest official sample demo data (`data/sample_demo.csv`) derived from official captures.
  - [ ] Retrain baseline and world model strictly on real chronological training splits.

---

## 10. Audit Sign-Off: What Was Inspected vs. What Remains Uncertain

### Inspected & Confirmed:
1. Every existing Python file (`src/contracts.py`, `src/model/baseline.py`, `src/model/world_model.py`, package `__init__.py` files).
2. Configuration and requirements (`requirements.txt`, `.gitignore`).
3. Serialized binary artifact (`artifacts/baseline_rf.pkl`) and its pickled internal structure.
4. Python environment, installed packages, and Windows AppLocker execution constraints.
5. All git commit diffs from repository inception.

### Remaining Uncertainties & External Blockers:
1. **Physical Dataset Availability:** The local `data/` directory contains zero files. Real training and evaluation cannot proceed until the operator provides raw CIC-IDS2017 or UNSW-NB15 files or downloads official archives.
2. **Scipy/Scikit-learn Execution Under WDAC:** Scikit-learn C-extensions (`_special_ufuncs.pyd`) are currently blocked by Windows Application Control policies inside `.venv`. Running baseline training/evaluation requires granting execution rights or running within an approved Python environment.
