# Task 3 Report — Defense Mechanism and Attack Simulation (FEDDEF)

**Author:** Kunal V Kurundwadkar (IIT2024177) — Task 3  
**Date:** 14 November 2026  
**Status:** Complete — 28 new tests, all passing (total suite: 94 passed, 4 skipped)  
**Source reference:** maskeensingh/FEDDEF (Colab prototype, `main.py` + `streamlit.py`)

---

## 1. Executive Summary

Task 3 delivers the **robust aggregation defense mechanism** (`src/defense.py`) and the **attack simulation module** (`src/attacks.py`) for the FedDef project.

The core defense is the **FEDDEF anomaly-scoring algorithm**, ported from the original Colab prototype in `maskeensingh/FEDDEF`. The algorithm detects Byzantine-faulty clients by measuring how much each client's predictions deviate from the cross-client consensus on a held-out validation set, then penalises or excludes anomalous clients before the weighted FedAvg aggregation step.

Key achievements:
- Implemented `src/defense.py`: `compute_anomaly_scores()`, `robust_aggregate()`, `run_defense()`, and full artifact persistence.
- Implemented `src/attacks.py`: 5 poisoning attack types (`random`, `sign_flip`, `scaling`, `label_flip`, `collusion`) with a `make_attack_fn()` factory.
- Added 28 new tests in `tests/test_defense.py` covering anomaly scoring, robust aggregation, all attack types, and end-to-end defense runs.
- Total test suite: **94 passed, 4 skipped** (skipped = real-dataset integration tests that require `framingham.csv`).

---

## 2. Source: maskeensingh/FEDDEF

The original prototype in `maskeensingh/FEDDEF` implemented the defense in a Colab notebook (`main.py`) and a Streamlit app (`streamlit.py`). The key prototype code blocks that this task formalises are:

**Anomaly detection (from FEDDEF `main.py`):**
```python
z0 = 1
theta = 0.25
client_confidence = {i: 0 for i in range(num_clients)}

for i in range(len(predictions)):
    pred_i = predictions[i]
    mu = pred_i.mean()
    stdd = pred_i.std()
    for j in range(len(predictions[i])):
        pred_i[j] = (pred_i[j] - mu) / stdd
        if abs(pred_i[j]) >= z0:
            client_confidence[j] += (abs(pred_i[j]) - z0)

# Normalise scores
sum_scores = sum(client_confidence.values())
for client in client_confidence:
    client_confidence[client] /= sum_scores
    if client_confidence[client] >= theta:
        len_data[client] = 0            # exclude
    else:
        len_data[client] *= (1 - client_confidence[client])  # penalise
```

**Attack simulation (from FEDDEF `main.py`):**
```python
if i in mal_client:
    coefs = np.random.rand(coefs.shape[0])
    intercept = np.random.randint(100000)
    beta = sigma / (l_r * data_i.shape[0])
    coefs *= beta
```

---

## 3. Implementation Details

### 3.1 `src/defense.py`

#### `compute_anomaly_scores(updates, X_val, z0, global_params)`

**Algorithm (per round):**

1. For each client $k$, compute the positive-class probability on each validation sample using the sigmoid of the client's logit: $p_{i,k} = \sigma(x_i^\top \theta_k + b_k)$.
2. For each validation sample $i$, standardise the prediction vector across clients:
   $$z_{i,k} = \frac{p_{i,k} - \mu_i}{\sigma_i}$$
3. Accumulate anomaly mass:
   $$s_k \mathrel{+}= \max(|z_{i,k}| - z_0,\ 0) \quad \forall i$$
4. Normalise to a probability simplex: $\hat{s}_k = s_k / \sum_j s_j$.

**Improvements over the prototype:**
- Uses direct sigmoid computation instead of `LinearRegression.predict()` — avoids R² on binary labels.
- Correctly handles zero standard deviation (all clients agree → no anomaly signal).
- Uses `predict_proba`-equivalent probabilities rather than raw regression predictions.

#### `robust_aggregate(updates, weights, X_val, z0, theta)`

After scoring, effective weights are computed as:

$$w_k^{\text{eff}} = \begin{cases} 0 & \text{if } \hat{s}_k \geq \theta \\ n_k \cdot (1 - \hat{s}_k) & \text{otherwise} \end{cases}$$

Then `fedavg(updates, effective_weights)` is called on the surviving clients.  
**Fallback:** if all clients are excluded, the original weights are restored to prevent collapse.

#### `run_defense(bundle, config, attack_fn, verbose, evaluate_test)`

Orchestrates the full defended FedAvg loop:
1. Partition training data via `src.federated.partition_dataset`.
2. Initialise global model on the full training set.
3. Per round: local `train_local_step` → optional attack → `robust_aggregate` → validation metrics.
4. Final test evaluation.

Saves artifacts under `results/defense/`:
- `defense_config.json`
- `defense_val_metrics.json`, `defense_test_metrics.json`
- `defense_test_confusion_matrix.png`
- `defense_convergence.csv`, `defense_convergence.png`
- `defense_exclusions.png` — bar chart of excluded clients per round
- `defense_comparison.json` — comparison with clean FedAvg baseline

### 3.2 `src/attacks.py`

| Attack | Description | Prototype origin |
|---|---|---|
| `random` | Replace params with scaled Gaussian noise | Directly matches FEDDEF prototype |
| `sign_flip` | Negate all parameters (Byzantine) | Standard FL adversarial benchmark |
| `scaling` | Amplify update magnitude | Norm-based bypass attempt |
| `label_flip` | Negate coef (simulates label inversion) | Standard FL adversarial benchmark |
| `collusion` | All malicious clients send same random vector | Coordination attack |

The `make_attack_fn(attack_type, strength)` factory returns a bound `(params, rng) → params` callable compatible with `run_defense`.

---

## 4. Test Suite (`tests/test_defense.py`)

28 new tests in 5 test classes:

| Class | Tests | What is verified |
|---|---|---|
| `TestComputeAnomalyScores` | 7 | Score shape, normalisation, identical clients → zero, malicious gets highest score, input validation |
| `TestRobustAggregate` | 5 | Output coef shape, honest clients never excluded, negative-weight validation, fallback when all excluded |
| `TestAttacks` | 8 | Each attack type, `make_attack_fn` factory, `list_attacks()`, unknown attack raises |
| `TestDefenseRun` | 7 | End-to-end clean and attacked runs, history length, required fields, coef shape, artifact creation |
| `TestDefenseVsAttack` | 1 | Defense runs correctly under attack |

Combined repository test status: **94 passed, 4 skipped** (skipped = `test_real_data.py` integration tests requiring `framingham.csv`).

---

## 5. CLI Usage

```bash
# Run defense under random noise attack (20% malicious, θ=0.25)
python3 -m src.defense --data data/framingham.csv --attack-type random

# Run clean defense (no attack) to verify no accuracy degradation
python3 -m src.defense --data data/framingham.csv --attack-type none

# Custom hyperparameters
python3 -m src.defense \
    --data data/framingham.csv \
    --attack-type sign_flip \
    --theta 0.3 \
    --z0 1.5 \
    --mal-fraction 0.3 \
    --epochs 50 \
    --results-dir results/defense
```

---

## 6. Algorithm Hyperparameters

| Hyperparameter | Default | Description |
|---|---|---|
| `z0` | 1.0 | Z-score threshold for anomaly accumulation |
| `theta` (θ) | 0.25 | Exclusion threshold; clients with normalised score ≥ θ are excluded |
| `mal_fraction` | 0.2 | Fraction of clients assumed to be malicious |
| `n_clients` | 20 | Total number of federated clients |

The prototype used `z0=1`, `theta=0.25` with 20 clients (4 malicious at 20% malicious fraction).

---

## 7. Artifact Manifest

Artifacts are saved under `results/defense/`:

```text
results/defense/
├── defense_config.json              # Full experiment configuration
├── defense_val_metrics.json         # Final validation metrics
├── defense_test_metrics.json        # Final test metrics (acc, f1, auc, etc.)
├── defense_test_confusion_matrix.png
├── defense_convergence.csv          # Per-round val metrics
├── defense_convergence.png          # 3-panel plot: Accuracy, F1, AUC
├── defense_exclusions.png           # Bar chart: clients excluded per round
└── defense_comparison.json         # Baseline vs defense metric comparison
```

---

## 8. Hand-off to Subsequent Tasks

- **For Task 4 (Experiments — Team Member 4):** Use `src.attacks.make_attack_fn` and `src.defense.run_defense` to build the full 3-setting experiment matrix (clean / attacked / defended). Reference the experiment matrix in `team_plan.md` Section 7.
- **For the Streamlit app:** A "Defense (Task 3)" tab should use `run_defense(bundle, config, attack_fn=make_attack_fn(attack_type))` and display `defense_convergence.png`, `defense_exclusions.png`, and the comparison metrics side-by-side with the baseline tab.
- **For the report:** The clean FedAvg baseline metrics in `results/baseline/fedavg_val_metrics.json` are the reference; the defense must not degrade clean accuracy by more than 1%.
