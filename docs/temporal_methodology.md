# NetworkSentinel AI — Temporal Evaluation Methodology & Windowing Architecture

**Problem Statement:** Smart India Hackathon 2026 Problem Statement 26153  
**Document:** `docs/temporal_methodology.md`  
**Date:** 2026-11-26  
**Implementation Status:** Active & Enforced in `src/pipeline/windowing.py`  

---

## 1. The Challenge of Temporal Network Forecasting

Traditional network intrusion detection evaluates machine learning models as independent, identically distributed (i.i.d.) classifiers on isolated network flows. In operational Security Operations Centers (SOCs), this assumption is deeply flawed:
1. **Temporal Autocorrelation:** Network packets and flows form coordinated sessions, reconnaissance probes, and multi-stage exploit campaigns.
2. **Reconnaissance-to-Exploitation Progression:** Adversaries do not attack at random; they progress through MITRE ATT&CK stages (Reconnaissance $\to$ Discovery $\to$ Exploits $\to$ Persistence $\to$ Impact).
3. **Severe Lookahead Leakage:** Shuffling network flows randomly across train and test sets leaks information from the same TCP session across both partitions, resulting in artificially near-perfect test accuracy ($>99\%$) that collapses when deployed on real live traffic.

NetworkSentinel AI replaces static flow classification with **chronological multi-step attack forecasting**.

---

## 2. Explicit Temporal Interval Formulation

Network traffic is ingested as an ordered chronological sequence of flow feature vectors:

$$S = \{(\mathbf{x}_1, y_1, s_1, t_1, d_1), (\mathbf{x}_2, y_2, s_2, t_2, d_2), \dots, (\mathbf{x}_T, y_T, s_T, t_T, d_T)\}$$

where:
- $\mathbf{x}_t \in \mathbb{R}^{D}$ ($D = 20$ standard network flow features).
- $y_t \in \{0, 1\}$ (Binary attack indicator: $0 = \text{Benign/Normal}, 1 = \text{Attack}$).
- $s_t \in \{0, 1, \dots, 9\}$ (Multiclass MITRE stage: Normal, Reconnaissance, DoS, Exploits, etc.).
- $t_t \in \mathbb{R}$ or `datetime` (Flow arrival / start timestamp).
- $d_t \in \mathbb{R}_{\ge 0}$ (Flow active duration). Flow completion time is $\tau_{\text{end}} = t_t + d_t$.

### 2.1 Window Dimensions & Explicit Intervals
For each forecasting window sample $i \in \{0, 1, \dots, M-1\}$ within a partition:
- **Lookback Input Interval:** $[i, i + W - 1]$ ($W = 10$).
  $$\mathbf{X}_{i} = [\mathbf{x}_i, \mathbf{x}_{i+1}, \dots, \mathbf{x}_{i+W-1}] \in \mathbb{R}^{W \times D}$$
  - *Input Time Range:* $[t_i, t_{i+W-1}]$.
  - *Forecast Issuance Time:* $t_{\text{pred}} = t_{i+W-1}$. Only information up to and including $t_{\text{pred}}$ is accessible.
- **Forecast Target Horizon Interval:** $[i + W, i + W + H - 1]$ ($H = 4$).
  $$\mathbf{y}^{\text{future}}_{i} = [y_{i+W}, y_{i+W+1}, \dots, y_{i+W+H-1}] \in [0, 1]^{H}$$
  - *Target Time Range:* $[t_{i+W}, t_{i+W+H-1}]$.
  - By construction: $\min(\text{Target Timestamps}) > \max(\text{Input Timestamps})$ for every valid window sample.
- **Immediate Next Stage Target:** $s_{i+W}$ (the MITRE stage at step $i + W$).

---

## 3. Partition Boundary Policy & Active Invariant Enforcement

```
Timeline: [========================================================================>]
Partitions:
[--- Training Split ---]        [Duration Gap]        [--- Validation ---]        [Duration Gap]        [--- Test ---]
(Max flow end: T_tr_end)  (T_tr_end < T_val_start)    (Max end: T_val_end)  (T_val_end < T_test_start)
```

### 3.1 Why Fixed Row Buffers are Insufficient
A fixed mechanical row buffer (e.g. 13 rows) was previously considered, but is **empirically insufficient** on real network traffic:
1. **Timestamp Burst Concurrency:** High-speed network taps record dozens or hundreds of flows sharing the exact same millisecond or second timestamp. Slicing by row count would bisect a concurrent flow burst across train and validation.
2. **Variable Flow Durations:** Network flows persist over time (`dur`). A flow starting in training at $t = 100\text{s}$ with duration $d = 60\text{s}$ continues actively until $t = 160\text{s}$. If the validation split begins at $t = 110\text{s}$, the training flow is actively communicating during the validation observation window!

### 3.2 The Implemented Policy (`chronological_split`)
`src/pipeline/windowing.py` implements an active, time-and-duration-aware partitioning algorithm:

1. **Pre-Condition Checks:**
   - DataFrame must be chronologically ordered (`df[ts_col].is_monotonic_increasing`).
   - Timestamps must not contain null/NaT values.
   - If timestamps are completely missing, the split fails loudly with `ValueError` unless explicitly running offline smoke tests with `allow_index_fallback=True`.
2. **Session / Duration Isolation:**
   - For every flow in the training partition, its completion time $t_{\text{end}} = t_{\text{start}} + \text{duration}$ is calculated.
   - The validation partition start time is enforced to satisfy:
     $$T_{\text{val\_start}} > \max_{i \in \text{Train}} (t_{\text{start}}^{(i)} + d^{(i)}) + \Delta_{\text{gap}}$$
   - Similarly, the test partition start time satisfies:
     $$T_{\text{test\_start}} > \max_{j \in \text{Val}} (t_{\text{start}}^{(j)} + d^{(j)}) + \Delta_{\text{gap}}$$
3. **Cluster Integrity for Duplicate Timestamps:**
   - If a boundary cutoff falls inside a cluster of duplicate timestamps $t^*$, the algorithm keeps all flows of $t^*$ together.
   - It verifies whether advancing to include the entire cluster maintains adequate size for subsequent partitions; if not, it cuts before $t^*$. Duplicate timestamps are never split across partition boundaries.
4. **Mandatory Production Verification:**
   - Every execution of `chronological_split()` invokes `verify_temporal_isolation()` before returning. If any flow completion time or timestamp overlaps across partitions, it raises an immediate `ValueError`.

---

## 4. Dataset Guarantees and Known Limitations

| Dataset | Available Temporal Fields | Policy Guarantees | Limitations & Risks |
| :--- | :--- | :--- | :--- |
| **CIC-IDS2017** | `timestamp` (1-second string resolution), `flow_duration` ($\mu\text{s}$) | Boundary duplicate clusters kept intact; flow durations converted to seconds ($\mu\text{s}/10^6$) to prevent session overlap across splits. | Flow start times have only 1-second resolution; inter-arrival times within the same second are determined by file arrival order. |
| **UNSW-NB15 (Raw 4 CSVs)** | `stime`, `ltime` (epoch float seconds), `dur` (seconds) | Sub-second start and end times allow exact session boundary exclusion; long-lived flows (>60s) are fully finished before subsequent splits begin. | Pre-split files (`UNSW_NB15_training-set.csv` and `testing-set.csv`) do NOT contain full chronological continuity and must not be concatenated. |
| **Sample Demo (`sample_demo.csv`)** | None (synthetic flow records) | Sliced via index fallback (`allow_index_fallback=True`) for UI layout smoke testing. | **Zero temporal isolation.** Synthetic demonstration data only; strictly prohibited as empirical research evidence under SIH 26153 rules. |
