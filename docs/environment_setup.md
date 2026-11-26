# NetworkSentinel AI — Environment Setup & Platform Compatibility Guide

**Problem Statement:** SIH 2026 Problem Statement 26153 (Defensive Network Traffic Analysis & Forecasting)  
**Document:** `docs/environment_setup.md`  
**Date:** 2026-11-26  

---

## 1. Executive Summary & Environment Diagnostic

During the foundation audit, the execution environment on Windows was diagnosed:
- **Operating System:** Windows 11 / Windows Server (AMD64)
- **Python Runtime:** Python 3.12.10
- **Virtual Environment:** `.venv/` configured with core packages
- **Working Libraries:**
  - `pandas==2.2.3` — Data manipulation & time-series operations (Operational)
  - `numpy==2.5.5` — Numerical vector and tensor math (Operational)
  - `torch==2.15.0` — PyTorch GRU, Attention, and Backpropagation (Operational)
  - `joblib==1.6.0` — Artifact serialization (Operational)
  - `streamlit==1.67.0` & `plotly==7.1.0` — SOC Dashboard components (Operational)
- **Windows Defender Application Control (WDAC) Restriction:**
  When importing `scikit-learn` or `scipy.special` in `.venv`, the Windows kernel blocks compiled C-extensions:
  ```
  ImportError: DLL load failed while importing _special_ufuncs: An Application Control policy has blocked this file.
  ```
  *Under SIH 26153 strict safety rules, WDAC security policies must not be disabled or weakened.*

This document details:
1. Native Windows installation and pipeline execution.
2. The standalone mathematical preprocessor designed to run without C-extension blocking.
3. WSL (Windows Subsystem for Linux) setup (Ubuntu / Kali Linux).
4. Google Colab execution for large-scale GPU training and benchmarking.

---

## 2. Windows Native Environment Setup

### 2.1 Prerequisites
- Python 3.10 to 3.12 (64-bit) installed.
- PowerShell 7 or Windows PowerShell.

### 2.2 Virtual Environment Creation & Package Installation
Open PowerShell in the repository root:

```powershell
# 1. Create a dedicated virtual environment if not already present
python -m venv .venv

# 2. Activate the virtual environment
.venv\Scripts\Activate.ps1

# 3. Upgrade pip and wheel
python -m pip install --upgrade pip setuptools wheel

# 4. Install project requirements
python -m pip install -r requirements.txt

# 5. Install pytest for test execution
python -m pip install pytest
```

### 2.3 WDAC Compatibility Strategy for Scikit-Learn
Because WDAC policy restricts unsigned or user-profile DLLs like `_special_ufuncs.pyd`:
1. **Pipeline Design:** All feature engineering, dataset validation, sliding-window generation, and robust standard scaling in `src/pipeline/` are implemented using native Python and vectorized NumPy/Pandas. They have **zero dependency on C-extensions from scipy/sklearn** and run natively in 100% of Windows environments.
2. **PyTorch World Model:** `src/model/world_model.py` uses PyTorch's native C++ runtime which is signed and permitted under WDAC.
3. **Baseline Model:** When running `BaselineAttackClassifier` on a machine with strict WDAC, run the baseline training via WSL or Google Colab (detailed below).

---

## 3. Windows Subsystem for Linux (WSL) Setup (Recommended for Full Baselines)

WSL bypasses Windows user-mode DLL execution restrictions because it runs a genuine Linux kernel with standard ELF binaries. WSL is already present on this workstation (Ubuntu 2, Kali Linux).

### 3.1 WSL Environment Initialization
Run in PowerShell:

```powershell
# Open default Ubuntu WSL instance
wsl -d Ubuntu
```

Inside WSL (bash):

```bash
# Update package lists and install build essentials & python3-venv
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv git

# Navigate to the Windows repository mount
cd /mnt/c/Users/Legion/sih-26153-attack-forecaster

# Create a Linux-native virtual environment
python3 -m venv .venv-linux
source .venv-linux/bin/activate

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt pytest

# Verify complete import suite including scikit-learn
python3 -c "import sklearn; import torch; import pandas; print('All dependencies operational in WSL!')"
```

---

## 4. Google Colab Setup (Free GPU / High-RAM Training)

For training the World Model on complete multi-gigabyte releases of CIC-IDS2017 or UNSW-NB15 with GPU acceleration:

1. Open a new notebook at [colab.research.google.com](https://colab.research.google.com).
2. Set Runtime type: **Runtime > Change runtime type > T4 GPU**.
3. Clone the repository and install dependencies:
   ```python
   !git clone -b development/repair https://github.com/ki6nn/sih-26153-attack-forecaster.git
   %cd sih-26153-attack-forecaster
   !pip install -r requirements.txt pytest
   ```
4. Run pipeline data validation and training:
   ```python
   !python -m unittest discover tests
   !python main.py
   ```

---

## 5. Requirements Specification & Dependency Rationale

The project specifies the following dependencies in `requirements.txt`:

```text
numpy>=1.26.0
pandas==2.2.3
scikit-learn>=1.4.0
torch>=2.2.0
shap>=0.45.0
streamlit>=1.35.0
plotly>=5.20.0
joblib>=1.3.0
pytest>=8.0.0
```

### Dependency Rationale:
- `pandas==2.2.3`: Strict pin for consistent DateTime parsing and rolling window aggregations across platforms.
- `numpy>=1.26.0`: Fast vectorized operations, window indexing, and tensor conversions.
- `torch>=2.2.0`: PyTorch tensor operations, 2-layer GRU, attention scoring, and multi-task loss calculation.
- `scikit-learn>=1.4.0`: Reference Random Forest baseline classifier and metrics.
- `joblib>=1.3.0`: Serialization of pipeline transformers and model state.
- `streamlit>=1.35.0` & `plotly>=5.20.0`: Real-time SOC dashboard interface for replaying traffic windows.
- `pytest>=8.0.0`: Automated test runner for unit and integration verification.
- `shap>=0.45.0`: Feature importance explanations.

---

## 6. Troubleshooting Common Issues

| Symptom | Cause | Solution |
| :--- | :--- | :--- |
| `ImportError: DLL load failed: An Application Control policy has blocked this file` | Windows Defender Application Control (WDAC) blocking `_special_ufuncs.pyd` in `.venv`. | Use WSL (`wsl -d Ubuntu`) for scikit-learn tasks. The pipeline modules (`src/pipeline/`) run natively on Windows using pure NumPy/Pandas. |
| `AttributeError: Can't get attribute 'BaselineAttackClassifier' on <module '__main__'>` | Model was pickled when running `baseline.py` as `__main__`. | Serialized model must be saved via module import (`from src.model.baseline import ...`) so class metadata binds to `src.model.baseline`. |
| `FileNotFoundError: data/CIC-IDS2017` or `data/UNSW-NB15` | Datasets have not been downloaded to `data/`. | Refer to `docs/datasets.md` for official download links and placement instructions. |
| `ValueError: y_true must contain binary labels 0 and 1` | Ground-truth labels contain multiclass strings. | Ensure `map_stage_labels()` or binary conversion (`y != 'Normal'`) is applied before evaluation. |
