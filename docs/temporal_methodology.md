# NetworkSentinel AI — Temporal Evaluation Methodology & Windowing Architecture

**Problem Statement:** Smart India Hackathon 2026 Problem Statement 26153  
**Document:** `docs/temporal_methodology.md`  
**Date:** 2026-11-26  

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

$$S = \{(\mathbf{x}_1, y_1, s_1, t_1), (\mathbf{x}_2, y_2, s_2, t_2), \dots, (\mathbf{x}_T, y_T, s_T, t_T)\}$$

where:
- $\mathbf{x}_t \in \mathbb{R}^{D}$ ($D = 20$ standard network flow features).
- $y_t \in \{0, 1\}$ (Binary attack indicator: $0 = \text{Benign/Normal}, 1 = \text{Attack}$).
- $s_t \in \{0, 1, \dots, 9\}$ (Multiclass MITRE stage: Normal, Reconnaissance, DoS, Exploits, etc.).
- $t_t \in \mathbb{R}$ or `datetime` (Flow arrival timestamp).

### 2.1 Window Dimensions & Explicit Intervals
For each forecasting window sample $i \in \{0, 1, \dots, M-1\}$:
- **Lookback Input Interval:** $[i, i + W - 1]$
  $$\mathbf{X}_{i} = [\mathbf{x}_i, \mathbf{x}_{i+1}, \dots, \mathbf{x}_{i+W-1}] \in \mathbb{R}^{W \times D} \quad (W = 10)$$
  - *Input Time Range:* $[t_i, t_{i+W-1}]$
  - *Forecast Issuance Time:* $t_{\text{pred}} = t_{i+W-1}$. Only information up to and including $t_{\text{pred}}$ is accessible.
- **Forecast Target Horizon Interval:** $[i + W, i + W + H - 1]$
  $$\mathbf{y}^{\text{future}}_{i} = [y_{i+W}, y_{i+W+1}, \dots, y_{i+W+H-1}] \in [0, 1]^{H} \quad (H = 4)$$
  - *Target Time Range:* $[t_{i+W}, t_{i+W+H-1}]$
- **Immediate Next Stage Target:** $s_{i+W}$ (the MITRE stage at step $i + W$).

---

## 3. Partition Boundary Invariants & Universal Leakage Prevention

```
Timeline: [========================================================================>]
Partitions:
[--- Training Split ---]  [Boundary Exclusion]  [--- Validation ---]  [Boundary Exclusion]  [--- Test ---]
(Targets end at T_tr)   (T_tr < T_val_in)       (Targets end T_val)  (T_val < T_test_in)
```

### 3.1 Why a Row-Count Buffer is NOT a Universal Guarantee
A fixed row-count buffer (e.g. 13 rows) is an internal mechanical safeguard for sequence index sliding, but **does not universally guarantee temporal non-overlap** in network traffic because:
1. **Timestamp Ties / Burst Concurrency:** High-speed network flows frequently share identical millisecond timestamps. If a row-count split falls inside a burst of flows sharing timestamp $t^*$, flows from the exact same timestamp cluster would exist in both partitions.
2. **Variable Flow Durations:** Flows have active durations (`dur`). A flow initiated at $t_a < t_{\text{split}}$ with duration $d$ remains active until $t_a + d > t_{\text{split}}$, leaking in-flight session dynamics into the evaluation partition.
3. **Uneven Event Distribution:** During quiet hours, 13 rows can span 2 hours; during a DDoS flood, 13 rows span 2 milliseconds.

### 3.2 Strict Mathematical Invariants
To guarantee rigorous empirical separation, the pipeline enforces two strict invariants:

$$\max_{j \in \mathcal{D}_{\text{train}}} \tau_{\text{target\_end}}^{(j)} < \min_{k \in \mathcal{D}_{\text{val}}} \tau_{\text{input\_start}}^{(k)}$$

$$\max_{k \in \mathcal{D}_{\text{val}}} \tau_{\text{target\_end}}^{(k)} < \min_{l \in \mathcal{D}_{\text{test}}} \tau_{\text{input\_start}}^{(l)}$$

That is:
- **Every training target interval must strictly precede every validation input interval.**
- **Every validation target interval must strictly precede every test input interval.**
- **Duplicate Boundary Timestamps are Blocked:** If duplicate timestamps exist at partition boundaries, `chronological_split(..., timestamp_col=...)` advances the boundary so that all events with timestamp $t \le t_{\text{boundary}}$ remain in the earlier partition, and the subsequent partition begins strictly at $t > t_{\text{boundary}}$.

---

## 4. Operational Limitations & Edge Cases

### 4.1 Single-Class Partitions
In datasets with sparse attack schedules (e.g. CIC-IDS2017 Monday is 100% benign), a training or validation split might contain zero positive attack labels. The pipeline:
- Avoids division by zero in metrics (`zero_division=0`).
- Provides explicit test fixtures validating that preprocessors and window generators execute gracefully on single-class inputs without crashing or producing NaNs.

### 4.2 Non-Monotonic Capture Files
Some network taps record packets out of arrival order due to multi-threaded queueing. The pipeline explicitly sorts by timestamp (`parse_timestamps(..., sort_chronological=True)`) and audits monotonicity in `generate_data_quality_report()`.

### 4.3 Deterministic Construction
Given a fixed dataset, feature column list, window size $W$, and horizon $H$, window construction is 100% deterministic, functional, and side-effect free.
