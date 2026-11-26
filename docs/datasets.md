# NetworkSentinel AI — Dataset Policy, Provenance, and Governance

**Problem Statement:** Smart India Hackathon 2026 Problem Statement 26153  
**Document:** `docs/datasets.md`  
**Date:** 2026-11-26  
**Audience:** Security ML Engineers, Evaluators, and Auditors  

---

## 1. Principles of Empirical Integrity

In defensive intrusion detection and attack forecasting research, empirical validity depends entirely on dataset fidelity. The following rules are binding across NetworkSentinel AI:

1. **No Synthetic Fallbacks for Evidence (Strict Rule 6):** Under no circumstances will synthetic, Gaussian, or randomized data be represented as research findings or benchmark results.
2. **No Timeline Fabrication (Strict Rule 7):** Pre-partitioned training and testing sets (e.g. `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv`) must **never be concatenated** to simulate a continuous temporal stream. These files represent post-hoc stratified samples; concatenating them introduces artificial time discontinuities and false temporal correlations.
3. **Primary Temporal Benchmark:** **CIC-IDS2017** is designated as the primary dataset for multi-step temporal attack forecasting due to its multi-day contiguous network capture design.
4. **Secondary Baseline / Detection Benchmark:** **UNSW-NB15** is designated for isolated detection experiments and MITRE stage mapping validation, or for temporal analysis only when using the un-partitioned continuous raw PCAP/CSV captures (`UNSW-NB15_1.csv` through `UNSW-NB15_4.csv`) sorted by flow start time (`stime`).

---

## 2. In-Repo Artifact Provenance: `data/sample_demo.csv`

An audit was conducted on `data/sample_demo.csv` present in the repository root:

- **File:** `data/sample_demo.csv`
- **SHA-256 Checksum:** `069a7866c467a8199016cdd288d87dd6bc0328c2e083710a5f98e8b584caa17a`
- **Provenance Classification:** **Synthetic demonstration data**
- **Derivation Procedure:** Generated via `src.pipeline.preprocessor._synthetic_dataset` (commit `8eb9f81`) using `np.random.default_rng(42)` across 5 stages (Normal, Reconnaissance, Exploits, Backdoor, DoS) with 100 rows per stage, producing 20 numeric flow columns without timestamps.
- **Empirical Status:** **NOT an official dataset extract.** It contains no genuine network packets or flows.
- **Permitted Use:** Strictly confined to offline SOC dashboard UI demonstration and smoke testing.
- **Prohibited Use:** Prohibited from being reported as research findings or benchmark evaluation evidence.
- **Machine-Readable Metadata:** Stored in `data/sample_demo_provenance.json`.

---

## 3. Official Dataset Sources & Provenance

### 3.1 CIC-IDS2017 (Canadian Institute for Cybersecurity)
- **Institution:** University of New Brunswick (UNB), Canada.
- **Official Portal:** [UNB CIC-IDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
- **Capture Window:** Monday, July 3, 2017 at 9:00 AM to Friday, July 7, 2017 at 5:00 PM (5 continuous days).
- **Network Environment:** Realistic B-Profile system generating naturalistic background benign traffic alongside executed attack campaigns (Brute Force, DoS, Heartbleed, Web Attacks, Infiltration, Botnet, PortScan, DDoS).
- **Official CSV Distribution:** Generated via CICFlowMeter, providing ~83 statistical flow features.
- **Licensing & Attribution:**
  > Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization.* Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP), Portugal.

### 3.2 UNSW-NB15 (Cyber Range Lab of the Australian Centre for Cyber Security)
- **Institution:** University of New South Wales (Canberra), Australian Defence Force Academy (ADFA).
- **Official Portal:** [UNSW-NB15 Dataset Repository](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- **Capture Window:** Captured using the Ixia PerfectStorm tool in two sessions:
  - Session 1: 16 hours on January 22, 2015.
  - Session 2: 15 hours on February 16, 2015.
- **Official Files:**
  - `UNSW-NB15_1.csv` through `UNSW-NB15_4.csv` (Raw chronological captures, 2,540,044 total flows).
  - `UNSW-NB15_LIST_EVENTS.csv` (Ground truth event logs).
  - `NUSW-NB15_features.csv` (Metadata schema).
- **Cautionary Note on Pre-split Files:** The files `UNSW_NB15_training-set.csv` (82,332 records) and `UNSW_NB15_testing-set.csv` (175,341 records) are pre-partitioned with synthetic class distributions. They must not be chained together.
- **Licensing & Attribution:**
  > Moustafa, N., & Slay, J. (2015). *UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set).* Military Communications and Information Systems Conference (MilCIS), Canberra, Australia.

---

## 4. Expected Directory Structure

Raw and processed dataset files must reside exclusively within `data/` and must remain untracked by Git (except for verified `sample_demo.csv` and its metadata):

```
sih-26153-attack-forecaster/
├── data/
│   ├── sample_demo.csv                     # Synthetic UI demo slice (SHA256: 069a7866...)
│   ├── sample_demo_provenance.json         # Provenance metadata record
│   ├── cic_ids2017/                        # Primary Temporal Dataset (Untracked)
│   │   ├── Monday-WorkingHours.pcap_ISCX.csv
│   │   ├── Tuesday-WorkingHours.pcap_ISCX.csv
│   │   ├── Wednesday-workingHours.pcap_ISCX.csv
│   │   ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
│   │   ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
│   │   ├── Friday-WorkingHours-Morning.pcap_ISCX.csv
│   │   ├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
│   │   └── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
│   └── unsw_nb15/                          # Secondary Baseline Dataset (Untracked)
│       ├── UNSW-NB15_1.csv
│       ├── UNSW-NB15_2.csv
│       ├── UNSW-NB15_3.csv
│       ├── UNSW-NB15_4.csv
│       └── NUSW-NB15_features.csv
```

---

## 5. Automated Data Validation Procedure

Before any dataset file is admitted into the training pipeline, it must pass validation via `src.pipeline.validation`:

```python
from src.pipeline.ingestion import load_dataset
from src.pipeline.validation import generate_data_quality_report, validate_required_columns
from src.pipeline.features import get_feature_columns

# 1. Ingest dataset with automated column normalization and timestamp parsing
df = load_dataset("data/cic_ids2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", dataset_name="cic_ids2017")

# 2. Enforce 20-feature contract
features = get_feature_columns("cic_ids2017")
validate_required_columns(df, features)

# 3. Generate quality assessment report
report = generate_data_quality_report(df, dataset_name="cic_ids2017", numeric_columns=features, label_col="label")
print(f"Total Rows: {report['total_rows']}")
print(f"Temporal Monotonicity: {report['temporal_coverage'].get('is_monotonic_increasing')}")
print(f"Missing / Infinite Features: {report['columns_with_missing_or_inf']}")
```

---

## 6. Research Governance Policies (Phase 1.6 Hardening)

### 6.1 Cryptographic Provenance & Evidence Verification
- **Strict Provenance Requirement:** In research mode (`is_research=True` or `enforce_provenance=True`), the pipeline requires a registered `DatasetProvenance` record.
- **Cryptographic Checksum Verification:** When `sha256` is provided, the actual SHA-256 hash of the ingested file is computed and compared against the registered hash. Tampered, incomplete, or corrupted downloads are rejected.
- **Verification vs. User Attribution:** The pipeline cryptographically verifies file integrity and hash equivalence against the provenance record. Origin URLs and author citations remain user-supplied metadata from the dataset publishers.
- **Rejection of Synthetic Demonstrations:** Datasets flagged with `is_research_evidence: false` or `provenance_type: "synthetic_demonstration_data"` (e.g. `data/sample_demo.csv`) are rejected with `ValueError` when entering research pipelines.

### 6.2 Fail-Closed Target Label Policy
- **Mandatory Labels:** Every partition must contain a ground truth label column (`attack_cat` or `label`). If missing, the pipeline raises a descriptive `ValueError`.
- **Zero Null Imputation:** Null, empty string, or NaN labels raise `ValueError`. The pipeline never silently imputes missing labels as benign (`Normal`).
- **Binary vs. Tactical Categorization:**
  * Binary classification evaluates whether network traffic is benign ($0$) or attack ($1$).
  * Multiclass MITRE stage mapping is a high-level heuristic categorization into coarse tactical phases (Reconnaissance, Exploits, DoS, etc.) for sequence forecasting. It **does NOT establish verified MITRE ATT&CK enterprise technique attributions**.
  * Unknown categories map to `Generic` (attack stage with $y_{\text{attack}} = 1$) or raise `ValueError` in strict mode; unknown labels are **NEVER mapped to Normal**.

### 6.3 Explicit Feature Validation & Training-Only Imputation
- **Anomaly Detection:** Features are scanned for NaNs, $+ \infty$, $-\infty$, and non-numeric values via `detect_feature_anomalies()`.
- **Training-Only Fitting:** Imputation values (`impute_values_`) are fitted exclusively on the training partition (using finite medians).
- **Out-of-Sample Transformation:** Validation and test partitions are transformed using the pre-fitted training imputation statistics. Validation/test values cannot mutate training statistics.
- **Preservation of Legitimate Zeros:** Valid numeric $0.0$ network values (e.g. zero packet loss, zero TCP flags) are preserved untouched.

### 6.4 Timestamp and Duration Policies
- **Explicit Dataset Specifications:** `unsw_nb15` uses `stime` (epoch seconds) and `dur` (seconds). `cic_ids2017` uses `timestamp` (string) and `flow_duration` ($\mu\text{s}$, converted to seconds by dividing by $10^6$).
- **Strict Timestamp Parsing:** In research mode, any unparseable timestamp or null record fails immediately with `ValueError`.
- **Duration Acknowledgment:** Datasets lacking duration columns cannot be assumed instantaneous (0-duration) in research mode without explicit user acknowledgment (`acknowledge_zero_duration=True`).

---

## 7. Data Tier Definitions Summary

| Data Tier | Location | Source | Permitted Usage |
| :--- | :--- | :--- | :--- |
| **Tier 1: Research Data** | `data/cic_ids2017/`, `data/unsw_nb15/` | Official UNB / UNSW archives | Benchmark publication, final evaluation. |
| **Tier 2: Demo Data** | `data/sample_demo.csv` | Synthetic generator (`_synthetic_dataset`) | Offline UI layout and smoke testing only. Never research evidence. |
| **Tier 3: Training Split** | Memory / Cached partition | Chronological first 70% | Fitting preprocessors, training neural nets & RF. |
| **Tier 4: Validation Split** | Memory / Cached partition | Chronological intermediate 15% | Hyperparameter tuning, early stopping, threshold calibration. |
| **Tier 5: Test Split** | Memory / Cached partition | Chronological final 15% | Strictly held-out final performance measurement. |
