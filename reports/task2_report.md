# Task 2 Report — Clean FedAvg Baseline and Centralized Reference

**Author:** Pratik Rakhecha (IIT2024176) — replacing Dewansh Khandelwal for Task 2  
**Date:** 14 November 2026  
**Status:** Complete and verified on the Framingham dataset  

---

## 1. Executive Summary

Task 2 delivers a clean, no-attack Federated Averaging (**FedAvg**) baseline for the FedDef project and establishes the centralized logistic-regression reference benchmark. The goal of this task is to prove that distributed local training under the standard FedAvg protocol closely reproduces the performance of centralized training on the full clinical dataset without requiring patient data pooling.

Key achievements:
- Implemented `src/federated.py` with seeded client partitioning, sample-weighted parameter aggregation (`fedavg()`), true iterative local optimization, and server orchestration.
- Decided and formalized the **True FedAvg** protocol (McMahan et al., 2017), where clients start each round from the broadcasted global parameters and perform local optimization steps matching the balanced log-loss objective.
- Trained a centralized reference `LogisticRegression(max_iter=1000, class_weight="balanced")` on the full training split (2,968 rows).
- Evaluated both models on the held-out test split (848 rows). The FedAvg model achieves **66.86% accuracy**, **0.3510 F1 score**, and **0.6992 ROC-AUC**, closely matching the centralized reference (**66.98% accuracy**, **0.3548 F1**, **0.6987 ROC-AUC**).
- **Tolerance verification:** The accuracy difference is **-0.12%** (-0.0012), easily satisfying the acceptance tolerance criterion of **±2–3%**.
- All baseline convergence curves, evaluation metrics, confusion matrices, and comparison tables are saved under `results/baseline/` and are fully reproducible from a single command (`python main.py --baseline`).
- Added 10 tests in `tests/test_federated.py`, bringing the total test suite to **70 passing tests**.

---

## 2. Federated Learning Protocol Decision

### Protocol Formulation: True FedAvg vs. One-Shot Averaging

The task deliverables require deciding and documenting the FL protocol:

1. **Preferred: True FedAvg (Iterative Local Steps)**
   - In each communication round $t \in \{1, \dots, T\}$, the server broadcasts global model parameters $w^{(t-1)} = (\theta^{(t-1)}, b^{(t-1)})$ to all participating clients $k \in \{1, \dots, K\}$.
   - Client $k$ initializes its local model with $w^{(t-1)}$ and performs $E$ local gradient descent steps on its private partition $\mathcal{D}_k$:
     $$\theta_k^{(t)} \leftarrow \theta_k^{(t)} - \eta \nabla_\theta \mathcal{L}_k(\theta, b)$$
     $$b_k^{(t)} \leftarrow b_k^{(t)} - \eta \nabla_b \mathcal{L}_k(\theta, b)$$
     where $\mathcal{L}_k$ is the class-balanced negative log-likelihood with L2 regularization:
     $$\mathcal{L}_k(\theta, b) = -\sum_{i \in \mathcal{D}_k} \frac{w_i}{\sum_j w_j} \left[ y_i \log \sigma(z_i) + (1 - y_i) \log(1 - \sigma(z_i)) \right] + \frac{\lambda_k}{2} \|\theta\|^2$$
     with $z_i = x_i^\top \theta + b$, $w_i = \text{balanced sample weight}$, and $\lambda_k = \frac{1}{C \cdot \sum_j w_j}$.
   - The server collects updates from all clients and aggregates them weighted by client sample counts $n_k = |\mathcal{D}_k|$:
     $$w^{(t)} = \sum_{k=1}^K \frac{n_k}{\sum_{j=1}^K n_j} w_k^{(t)}$$
   - **Decision:** **True FedAvg is chosen as the primary protocol.** It directly matches the project proposal hyperparameters ($T = 50$ rounds, $\eta = 0.2$, $K = 20$ clients) and exhibits stable, smooth convergence.

2. **Alternative: One-Shot Averaging**
   - Each client independently fits a standard `LogisticRegression(max_iter=1000, class_weight="balanced")` to convergence on its local shard, and the server aggregates the resulting weight vectors once via `fedavg(updates, weights)`.
   - **Empirical comparison:** On our 20-client partition, one-shot averaging yields test accuracy of 73.23% but recall drops from 59.69% to 49.61% due to independent local regularization shifts on small sample sizes ($\approx 148$ rows/client). This produces an accuracy discrepancy of +6.25%, exceeding the ±3% tolerance.
   - **Conclusion:** While one-shot averaging is implemented in `src/federated.py` (`protocol="one_shot"`) as an experimental option, **True FedAvg is the superior, mathematically principled protocol** that faithfully mirrors centralized empirical risk minimization.

---

## 3. Implementation Details (`src/federated.py`)

### Parameter Representation
Adhering to the frozen team contract in `team_plan.md`:
```python
Params = {
    "coef": np.ndarray,       # 1D array of shape (n_features,), float64
    "intercept": float,      # scalar float
}
```

### Data Partitioning (`partition_dataset`)
- Partitions the training set ($N = 2,968$) across $K = 20$ clients ($\approx 148$ samples each).
- **Guarantees:**
  - Complete coverage: $\bigcup_{k=1}^K \mathcal{D}_k = \mathcal{D}_{\text{train}}$, with $\mathcal{D}_j \cap \mathcal{D}_k = \emptyset$ for $j \ne k$.
  - Seeded determinism: `np.random.default_rng(seed)`.
  - Class stratification: Stratified splitting ensures every client shard contains both positive ($y=1$) and negative ($y=0$) cases, preventing single-class failure in local logistic loss computation.

### Aggregation Function (`fedavg`)
- Mathematically validates all client updates: verifies 1D coefficient vector shapes, non-negativity of weights, finite values, and non-zero total weight.
- Computes exact sample-weighted linear combination:
  $$\bar{\theta} = \sum_{k=1}^K \alpha_k \theta_k, \quad \bar{b} = \sum_{k=1}^K \alpha_k b_k, \quad \alpha_k = \frac{n_k}{\sum_{j} n_j}$$
- If identical models are passed, `fedavg` returns that exact model regardless of weights.

### Server Loop (`run_federated`)
- Tracks convergence round-by-round by evaluating the global model on the validation split ($N_{\text{val}} = 424$).
- Records round index, validation accuracy, precision, recall, F1, and ROC-AUC.
- Evaluates the final model on the held-out test split ($N_{\text{test}} = 848$).

---

## 4. Experimental Results and Tolerance Verification

### Baseline Comparison on Held-Out Test Set (Seed = 42)

The table below presents the primary benchmark comparing the centralized reference model against the clean FedAvg baseline on the untouched test split (848 patients):

| Metric | Centralized Reference | Clean FedAvg Baseline | Absolute Difference | Relative Delta |
|---|---:|---:|---:|---:|
| **Accuracy** | **0.6698** (66.98%) | **0.6686** (66.86%) | **-0.0012** (-0.12%) | -0.18% |
| **Precision** | 0.2525 | 0.2500 | -0.0025 | -0.99% |
| **Recall** | 0.5969 | 0.5891 | -0.0078 | -1.31% |
| **F1 Score** | **0.3548** | **0.3510** | **-0.0038** | -1.07% |
| **ROC-AUC** | **0.6987** | **0.6992** | **+0.0004** | +0.06% |

### Confusion Matrices (Test Set, N = 848)

| Model | True Negatives (0,0) | False Positives (0,1) | False Negatives (1,0) | True Positives (1,1) |
|---|---:|---:|---:|---:|
| **Centralized** | 491 | 228 | 52 | 77 |
| **FedAvg** | 491 | 228 | 53 | 76 |

The confusion matrices are virtually indistinguishable: only a single positive sample (out of 129 test positives) differed in prediction between the centralized model and the 20-client federated model.

### Acceptance Criteria Verification Checklist

- [x] **Baseline accuracy, F1, and ROC-AUC reported on held-out test set:** Reported above.
- [x] **FedAvg is within documented tolerance of centralized model (±2–3% accuracy):**
  $$\text{Actual accuracy difference} = |0.6686 - 0.6698| = 0.0012 = 0.12\% \ll 3.0\% \quad \text{[PASSED]}$$
- [x] **Baseline curve is reproducible from one command:** Executed via `python main.py --baseline` or `python -m src.federated`.
- [x] **No attacker code is involved:** Clean baseline implementation only; attack hooks deferred to Task 4.

---

## 5. Convergence Analysis

The federated model was trained over 50 communication rounds ($K=20$ clients, $E=5$ local steps, $\eta=0.2$). Validation metrics evolved as follows:

| Round | Validation Accuracy | Validation F1 | Validation ROC-AUC |
|---:|---:|---:|---:|
| 1 | 0.6038 | 0.3438 | 0.7154 |
| 10 | 0.6226 | 0.3496 | 0.7210 |
| 20 | 0.6344 | 0.3515 | 0.7169 |
| 30 | 0.6415 | 0.3559 | 0.7153 |
| 40 | 0.6439 | 0.3574 | 0.7141 |
| 50 | **0.6415** | **0.3559** | **0.7139** |

*Centralized validation reference: Accuracy 0.6392, F1 0.3544, ROC-AUC 0.7120.*

The convergence curve demonstrates that:
1. Within 10 rounds, the model rapidly learns the dominant risk factors (age, blood pressure, smoking).
2. By round 30, validation accuracy and F1 stabilize within 0.2% of the centralized validation reference.
3. Over rounds 30–50, the model reaches asymptotic convergence without overfitting or divergence.

The convergence plot is generated automatically and saved to `results/baseline/convergence.png`.

---

## 6. Multi-Seed Robustness

To ensure findings are not artifacts of a single random split, we evaluated FedAvg against the centralized benchmark across three independent seeds (42, 43, and 44):

| Seed | Model | Test Accuracy | Test F1 | Test ROC-AUC | Accuracy Difference | Status |
|---|---|---:|---:|---:|---:|---|
| **42** | Centralized | 0.6698 | 0.3548 | 0.6987 | — | — |
| | FedAvg | 0.6686 | 0.3510 | 0.6992 | **-0.0012** (-0.12%) | PASSED |
| **43** | Centralized | 0.6580 | 0.3498 | 0.6985 | — | — |
| | FedAvg | 0.6698 | 0.3636 | 0.6986 | **+0.0118** (+1.18%) | PASSED |
| **44** | Centralized | 0.6639 | 0.3709 | 0.7147 | — | — |
| | FedAvg | 0.6733 | 0.3775 | 0.7185 | **+0.0094** (+0.94%) | PASSED |
| **Mean ± Std** | **Centralized** | **0.6639 ± 0.0059** | **0.3585 ± 0.0110** | **0.7040 ± 0.0093** | — | — |
| | **FedAvg** | **0.6706 ± 0.0024** | **0.3640 ± 0.0133** | **0.7054 ± 0.0113** | **+0.0067 ± 0.0069** | **ALL PASSED** |

Across all three seeds, the accuracy difference between FedAvg and the centralized reference is within ±1.2%, significantly inside the ±3.0% acceptance boundary.

---

## 7. Artifact Manifest

All artifacts are persisted in version-controlled directory structures under `results/baseline/`:

```text
results/baseline/
├── baseline_config.json                  # Complete run configuration
├── centralized_test_confusion_matrix.png # Centralized confusion matrix heatmap
├── centralized_test_metrics.json         # Centralized test metrics (acc, f1, auc, etc.)
├── centralized_val_metrics.json          # Centralized validation metrics
├── comparison.json                       # Centralized vs FedAvg comparison & tolerance check
├── convergence.csv                       # Per-round validation metrics across 50 rounds
├── convergence.png                       # 3-panel convergence plot (Acc, F1, AUC)
├── fedavg_test_confusion_matrix.png      # FedAvg confusion matrix heatmap
├── fedavg_test_metrics.json              # FedAvg test metrics
└── fedavg_val_metrics.json               # FedAvg validation metrics
```

---

## 8. Test Suite Summary

The test suite in `tests/test_federated.py` verifies:
1. `test_weighted_average_correctness`: Analytic verification of `fedavg` on uniform, weighted, and asymmetric vectors.
2. `test_client_partition_covers_all_rows`: Complete disjoint partition, 100% row coverage, aligned indices, and positive/negative class presence across 2, 5, 10, and 20 clients.
3. `test_identical_models_average_to_themselves`: Zero variance when identical models are aggregated.
4. `test_same_seed_reproduces_the_same_result`: Bit-for-bit determinism across independent runs.
5. `test_different_seed_produces_different_partitions`: Verifies seed responsiveness.
6. `test_fedavg_input_validation`: Error handling for empty updates, mismatched dimensions, negative weights, zero-sum weights, and missing keys.
7. `test_create_model_from_params_and_vectorized_predict`: Verification that model wrapper and vectorized prediction produce identical outputs to scikit-learn.
8. `test_train_centralized_model_returns_valid_metrics`: Range and schema validation for reference model outputs.
9. `test_fedavg_is_within_tolerance_of_centralized`: Automated verification that accuracy difference is $\le 3.0\%$.
10. `test_federated_cli_end_to_end`: Subprocess execution of the baseline command verifying exit code 0, console output, and generation of all 10 artifact files.

Combined repository test status: **70 passed in 10.06s**.

---

## 9. Streamlit App Integration

The `Baseline (Task 2)` tab in `app.py` has been updated:
- Authorship credited to Pratik Rakhecha.
- Interactive controls for client count, rounds, learning rate, and FL protocol.
- Live execution button ("Run FedAvg Baseline Live") with real-time status.
- Side-by-side metric comparison cards with delta indicators against the centralized reference.
- Acceptance criteria tolerance badge (displays green checkmark when within ±3.0%).
- Inline rendering of the convergence curve (`convergence.png`) and confusion matrices.

---

## 10. Hand-off to Subsequent Tasks

- **For Task 3 (Defense — Kunal V Kurundwadkar):**
  - Use `src/federated.py` as the baseline aggregation loop.
  - Inject client anomaly scoring into the server aggregation step: before calling `fedavg(updates, weights)`, score `updates` against `bundle.X_val` and adjust `weights`.
  - The clean FedAvg baseline metrics in `results/baseline/fedavg_val_metrics.json` serve as the clean reference that the defended model must not degrade by more than 1%.
- **For Task 4 (Attacks — Team Member 4):**
  - Implement poisoning functions in `src/attacks.py` that take legitimate client parameter dictionaries (`{"coef": ..., "intercept": ...}`) and corrupt them before submission to the server loop.
  - The clean convergence curve in `results/baseline/convergence.csv` provides the benchmark curve to compare against under attack.
