"""Defense module for FedDef (Task 3).

Implements the FEDDEF anomaly-scoring robust aggregation mechanism derived
from the original prototype in maskeensingh/FEDDEF.

Algorithm (per communication round):
1. Each client submits a parameter update (coef + intercept).
2. The server predicts on the shared validation set using each client's model.
3. For every validation sample i, predictions from all K clients form a
   vector p_i ∈ R^K.  The z-score of each client's prediction is computed:
       z_{i,k} = (p_{i,k} - µ_i) / σ_i
4. A client's anomaly score is accumulated whenever |z_{i,k}| ≥ z0:
       confidence[k] += |z_{i,k}| - z0
5. Scores are normalised to sum to 1 (probability simplex).
6. A client whose normalised score ≥ theta is **excluded** (weight set to 0).
   All other clients have their FedAvg weight penalised by (1 - score):
       effective_weight[k] = n_k × (1 - confidence[k])
   if score < theta, else 0.
7. Standard weighted FedAvg aggregation is applied to the surviving updates.

This matches the defence implemented in the FEDDEF Colab notebook with the
following improvements:
- Uses the frozen team Params contract {"coef": np.ndarray, "intercept": float}.
- Uses the proper validation set (not the test set) for anomaly detection.
- Correctly uses LogisticRegression predict_proba scores instead of raw
  LinearRegression regression values.
- Integrates with src.federated.fedavg and src.evaluate.

Author: Kunal V Kurundwadkar (IIT2024177) — Task 3
Date: 14 November 2026
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config import Config
from src.data import DataBundle, load_data
from src.evaluate import (
    evaluate_model,
    format_metrics,
    save_confusion_matrix_plot,
    save_metrics,
)
from src.federated import (
    ClientDataset,
    FederatedResult,
    fedavg,
    partition_dataset,
    run_federated,
    train_local_step,
)


# ---------------------------------------------------------------------------
# Core defence primitives
# ---------------------------------------------------------------------------


def compute_anomaly_scores(
    updates: Sequence[Mapping[str, Any]],
    X_val: pd.DataFrame,
    *,
    z0: float = 1.0,
    global_params: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Compute per-client anomaly scores using z-score prediction deviation.

    For every validation sample i and each client k, construct a predicted
    probability vector p_i ∈ R^K.  Standardise across clients (z-score), and
    accumulate anomaly mass when |z_{i,k}| ≥ z0.

    Parameters
    ----------
    updates:
        List of K client parameter dicts {"coef": ndarray, "intercept": float}.
    X_val:
        Validation feature matrix (DataFrame), *not* scaled — it is assumed
        the caller passes the already-scaled validation set from DataBundle.
    z0:
        Anomaly threshold; default 1.0 (one standard deviation).
    global_params:
        Current global model parameters.  When provided, deviations are
        computed relative to the global model; otherwise raw per-sample
        probabilities are used directly.

    Returns
    -------
    scores : np.ndarray of shape (K,)
        Normalised anomaly scores summing to 1.  Clients with score near 0
        are trusted; clients near 1 are anomalous.
    """
    if len(updates) == 0:
        raise ValueError("updates must be non-empty")
    if X_val.empty:
        raise ValueError("X_val must be non-empty")
    if z0 < 0.0:
        raise ValueError("z0 must be non-negative")

    n_clients = len(updates)
    n_val = len(X_val)
    X_arr = X_val.values.astype(float)

    # Build per-client prediction probability arrays (positive-class prob)
    # Use direct sigmoid computation to avoid any sklearn fitting dependency.
    all_preds = np.zeros((n_val, n_clients), dtype=float)
    for k, params in enumerate(updates):
        coef = np.asarray(params["coef"], dtype=float).ravel()
        intercept = float(params["intercept"])
        logits = X_arr @ coef + intercept
        all_preds[:, k] = 1.0 / (1.0 + np.exp(-np.clip(logits, -35.0, 35.0)))

    # For each validation sample, compute cross-client z-scores and accumulate
    client_confidence = np.zeros(n_clients, dtype=float)
    for i in range(n_val):
        p_i = all_preds[i]  # shape (K,)
        mu = p_i.mean()
        sigma = p_i.std()
        if sigma < 1e-12:
            # All clients agree — no anomaly signal
            continue
        z_i = (p_i - mu) / sigma
        excess = np.abs(z_i) - z0
        client_confidence += np.where(excess > 0, excess, 0.0)

    # Normalise to a probability simplex
    total = client_confidence.sum()
    if total < 1e-12:
        # No anomalies detected: uniform trust
        return np.zeros(n_clients, dtype=float)

    return client_confidence / total


def robust_aggregate(
    updates: Sequence[Mapping[str, Any]],
    weights: Sequence[float] | np.ndarray,
    X_val: pd.DataFrame,
    *,
    z0: float = 1.0,
    theta: float = 0.25,
    global_params: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """FEDDEF robust aggregation: score → penalise/exclude → weighted FedAvg.

    Parameters
    ----------
    updates:
        K client parameter dicts.
    weights:
        K non-negative sample counts (or importance weights).
    X_val:
        Scaled validation features for anomaly detection.
    z0:
        Z-score threshold for anomaly accumulation.
    theta:
        Exclusion threshold; clients with normalised score ≥ theta are excluded.
    global_params:
        Current global parameters (passed through to anomaly scorer).

    Returns
    -------
    aggregated : dict
        Aggregated global parameters after defence.
    info : dict
        Diagnostic information: anomaly scores, effective weights, excluded mask.
    """
    weights_arr = np.asarray(weights, dtype=float)
    if (weights_arr < 0).any():
        raise ValueError("All weights must be non-negative")

    scores = compute_anomaly_scores(
        updates, X_val, z0=z0, global_params=global_params
    )

    # Compute effective weights
    effective_weights = np.zeros_like(weights_arr)
    excluded = np.zeros(len(updates), dtype=bool)
    for k in range(len(updates)):
        if scores[k] >= theta:
            excluded[k] = True
            effective_weights[k] = 0.0
        else:
            effective_weights[k] = weights_arr[k] * (1.0 - scores[k])

    # At least one client must survive
    if effective_weights.sum() < 1e-12:
        # Fallback: use all clients with original weights (trust nobody heuristic)
        effective_weights = weights_arr.copy()
        excluded[:] = False

    aggregated = fedavg(updates, effective_weights)

    info: dict[str, Any] = {
        "anomaly_scores": scores.tolist(),
        "effective_weights": effective_weights.tolist(),
        "excluded": excluded.tolist(),
        "n_excluded": int(excluded.sum()),
        "z0": z0,
        "theta": theta,
    }
    return aggregated, info


# ---------------------------------------------------------------------------
# Defense experiment runner
# ---------------------------------------------------------------------------


@dataclass
class DefenseResult:
    """Full summary of a defended federated run."""

    global_params: dict[str, Any]
    history: list[dict[str, Any]]
    final_val_metrics: dict[str, Any]
    final_test_metrics: dict[str, Any]
    config: dict[str, Any]
    n_excluded_per_round: list[int] = field(default_factory=list)
    n_clients: int = 0


def run_defense(
    bundle: DataBundle,
    config: Config,
    *,
    n_mal_clients: int | None = None,
    attack_fn: Any = None,
    verbose: bool = False,
    evaluate_test: bool = True,
) -> DefenseResult:
    """Run the FEDDEF defended FedAvg loop.

    Parameters
    ----------
    bundle:
        DataBundle from src.data.load_data.
    config:
        Shared Config (uses config.n_clients, epochs, lr, theta, z0,
        mal_fraction, seed).
    n_mal_clients:
        Override the number of malicious clients; defaults to
        int(config.n_clients * config.mal_fraction).
    attack_fn:
        A callable(params, rng) -> params that corrupts a client's update.
        When None, no attack is applied (clean defence run).
    verbose:
        Print per-round progress to stdout.
    evaluate_test:
        Whether to evaluate the final model on the test set.

    Returns
    -------
    DefenseResult with per-round history and final metrics.
    """
    rng = np.random.default_rng(config.seed)
    n_clients = config.n_clients
    n_mal = (
        n_mal_clients
        if n_mal_clients is not None
        else int(n_clients * config.mal_fraction)
    )
    mal_client_ids = set(range(n_mal))

    # Partition training data
    partitions: list[ClientDataset] = partition_dataset(
        bundle.X_train, bundle.y_train, n_clients=n_clients, seed=config.seed
    )

    # Initialise global params from first client's data (matches prototype)
    init_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=config.seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        init_model.fit(bundle.X_train, bundle.y_train)

    global_params: dict[str, Any] = {
        "coef": init_model.coef_.ravel().copy(),
        "intercept": float(init_model.intercept_[0]),
    }

    history: list[dict[str, Any]] = []
    n_excluded_per_round: list[int] = []

    for epoch in range(config.epochs):
        updates: list[dict[str, Any]] = []
        sample_weights: list[float] = []

        for client in partitions:
            cid = client.client_id
            # Local training (True FedAvg: start from global params)
            local_params = train_local_step(
                client.X,
                client.y,
                initial_params=global_params,
                lr=config.lr,
                local_epochs=5,
            )
            # Apply attack to malicious clients
            if attack_fn is not None and cid in mal_client_ids:
                local_params = attack_fn(local_params, rng)

            updates.append(local_params)
            sample_weights.append(float(client.n_samples))

        # Robust aggregation
        global_params, agg_info = robust_aggregate(
            updates,
            sample_weights,
            bundle.X_val,
            z0=config.z0,
            theta=config.theta,
            global_params=global_params,
        )
        n_excluded = agg_info["n_excluded"]
        n_excluded_per_round.append(n_excluded)

        # Evaluate on validation set
        tmp_model = _params_to_model(global_params, bundle.X_train)
        val_metrics = evaluate_model(
            tmp_model, bundle.X_val, bundle.y_val
        )

        round_info: dict[str, Any] = {
            "epoch": epoch + 1,
            "n_excluded": n_excluded,
            **{f"val_{k}": v for k, v in val_metrics.items() if k != "confusion_matrix"},
        }
        history.append(round_info)

        if verbose:
            print(
                f"Round {epoch+1:3d}/{config.epochs} | "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"val_f1={val_metrics['f1']:.4f} "
                f"val_auc={val_metrics.get('roc_auc', float('nan')):.4f} "
                f"excluded={n_excluded}/{n_clients}",
                flush=True,
            )

    # Final evaluation
    final_model = _params_to_model(global_params, bundle.X_train)
    final_val_metrics = evaluate_model(
        final_model, bundle.X_val, bundle.y_val
    )
    final_test_metrics: dict[str, Any] = {}
    if evaluate_test:
        final_test_metrics = evaluate_model(
            final_model, bundle.X_test, bundle.y_test
        )

    return DefenseResult(
        global_params=global_params,
        history=history,
        final_val_metrics=final_val_metrics,
        final_test_metrics=final_test_metrics,
        config=config.to_dict(),
        n_excluded_per_round=n_excluded_per_round,
        n_clients=n_clients,
    )


# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------


def save_defense_artifacts(
    result: DefenseResult,
    baseline_val_metrics: dict[str, Any],
    baseline_test_metrics: dict[str, Any],
    *,
    results_dir: Path,
) -> dict[str, Path]:
    """Persist all Task 3 defense artifacts.

    Returns a dict mapping artifact name → saved Path.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    # Configuration
    cfg_path = results_dir / "defense_config.json"
    with cfg_path.open("w", encoding="utf-8") as fh:
        json.dump(result.config, fh, indent=2)
    saved["defense_config"] = cfg_path

    # Val/test metrics
    val_path = save_metrics(result.final_val_metrics, results_dir / "defense_val_metrics.json")
    saved["defense_val_metrics"] = val_path

    if result.final_test_metrics:
        test_path = save_metrics(result.final_test_metrics, results_dir / "defense_test_metrics.json")
        saved["defense_test_metrics"] = test_path

    # Confusion matrix
    if result.final_test_metrics:
        cm_path = save_confusion_matrix_plot(
            result.final_test_metrics,
            results_dir / "defense_test_confusion_matrix.png",
            title="Defense: Confusion Matrix (Test Set)",
        )
        saved["defense_confusion_matrix"] = cm_path

    # Convergence CSV
    df = pd.DataFrame(result.history)
    conv_path = results_dir / "defense_convergence.csv"
    df.to_csv(conv_path, index=False)
    saved["defense_convergence"] = conv_path

    # Comparison JSON
    comparison = {
        "baseline_val": {k: v for k, v in baseline_val_metrics.items() if k != "confusion_matrix"},
        "defense_val": {k: v for k, v in result.final_val_metrics.items() if k != "confusion_matrix"},
        "baseline_test": {k: v for k, v in baseline_test_metrics.items() if k != "confusion_matrix"},
        "defense_test": {k: v for k, v in result.final_test_metrics.items() if k != "confusion_matrix"},
        "accuracy_improvement_vs_attacked": "see reports/task3_report.md",
        "n_excluded_per_round": result.n_excluded_per_round,
    }
    cmp_path = results_dir / "defense_comparison.json"
    with cmp_path.open("w", encoding="utf-8") as fh:
        json.dump(comparison, fh, indent=2)
    saved["defense_comparison"] = cmp_path

    # Convergence plot (3-panel: Accuracy, F1, AUC)
    conv_plot_path = _save_convergence_plot(df, results_dir / "defense_convergence.png")
    saved["defense_convergence_plot"] = conv_plot_path

    # Exclusion plot
    excl_plot_path = _save_exclusion_plot(result.n_excluded_per_round, results_dir / "defense_exclusions.png")
    saved["defense_exclusion_plot"] = excl_plot_path

    return saved


def _params_to_model(params: Mapping[str, Any], X_ref: pd.DataFrame) -> LogisticRegression:
    """Reconstruct a scikit-learn LogisticRegression from stored params."""
    from src.federated import create_model_from_params
    return create_model_from_params(params)


def _save_convergence_plot(df: pd.DataFrame, path: Path) -> Path:
    """Save a 3-panel convergence plot (Accuracy, F1, ROC-AUC)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics = [
        ("val_accuracy", "Validation Accuracy"),
        ("val_f1", "Validation F1"),
        ("val_roc_auc", "Validation ROC-AUC"),
    ]
    for ax, (col, label) in zip(axes, metrics):
        if col in df.columns:
            ax.plot(df["epoch"], df[col], linewidth=1.5, color="steelblue")
            ax.set_xlabel("Round")
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.grid(alpha=0.3)
    fig.suptitle("Defense Convergence (Task 3 — FEDDEF)", y=1.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_exclusion_plot(n_excluded: list[int], path: Path) -> Path:
    """Save a bar chart showing how many clients were excluded each round."""
    fig, ax = plt.subplots(figsize=(10, 3))
    rounds = list(range(1, len(n_excluded) + 1))
    ax.bar(rounds, n_excluded, color="tomato", width=0.8, alpha=0.8)
    ax.set_xlabel("Round")
    ax.set_ylabel("Clients excluded")
    ax.set_title("Clients excluded by FEDDEF defence per round")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def build_parser() -> "argparse.ArgumentParser":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the FEDDEF defence experiment (Task 3).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        metavar="CSV",
        help="Path to framingham.csv",
    )
    parser.add_argument(
        "--n-clients",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--mal-fraction",
        type=float,
        default=0.2,
        help="Fraction of malicious clients (default: 0.2 = 4 out of 20).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=0.25,
        help="Exclusion threshold for anomaly score (default: 0.25).",
    )
    parser.add_argument(
        "--z0",
        type=float,
        default=1.0,
        help="Z-score anomaly accumulation threshold (default: 1.0).",
    )
    parser.add_argument(
        "--attack-type",
        type=str,
        default="random",
        choices=["random", "sign_flip", "scaling", "none"],
        help="Attack type to run before defence.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/defense"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    import argparse
    from src.attacks import make_attack_fn

    parser = build_parser()
    args = parser.parse_args(argv)

    # Resolve dataset
    data_path = args.data
    if data_path is None:
        for candidate in (Path("data/framingham.csv"), Path("framingham.csv")):
            if candidate.exists():
                data_path = candidate
                break
        if data_path is None:
            data_path = Path("data/framingham.csv")

    if not data_path.exists():
        print(f"Dataset not found at {data_path}. Please place framingham.csv in data/.")
        return 1

    config = Config(
        seed=args.seed,
        n_clients=args.n_clients,
        mal_fraction=args.mal_fraction,
        epochs=args.epochs,
        lr=args.lr,
        theta=args.theta,
        z0=args.z0,
        attack_type=args.attack_type if args.attack_type != "none" else "random",
        data_path=data_path,
        results_dir=args.results_dir.parent,
    )

    print("\n[1/4] Loading Framingham dataset...")
    bundle = load_data(
        config.data_path,
        target_col=config.target_col,
        test_size=config.test_size,
        val_size=config.val_size,
        random_state=config.seed,
    )
    print(
        f"Loaded: {bundle.n_train} train / {bundle.n_val} val / "
        f"{bundle.n_test} test rows, {bundle.n_features} features."
    )

    print("\n[2/4] Loading clean FedAvg baseline metrics...")
    baseline_val_path = Path("results/baseline/fedavg_val_metrics.json")
    baseline_test_path = Path("results/baseline/fedavg_test_metrics.json")
    if baseline_val_path.exists():
        from src.evaluate import load_metrics
        baseline_val = load_metrics(baseline_val_path)
        baseline_test = load_metrics(baseline_test_path) if baseline_test_path.exists() else {}
        print(f"  Baseline val accuracy: {baseline_val.get('accuracy', 'n/a'):.4f}")
    else:
        print("  Baseline metrics not found — run `python main.py --baseline` first.")
        baseline_val = {}
        baseline_test = {}

    # Build attack function
    attack_fn = None if args.attack_type == "none" else make_attack_fn(args.attack_type, strength=1.0)

    print(
        f"\n[3/4] Running FEDDEF defence "
        f"(attack={args.attack_type}, θ={args.theta}, z₀={args.z0}, "
        f"malicious={int(args.n_clients * args.mal_fraction)}/{args.n_clients})..."
    )
    result = run_defense(
        bundle,
        config=config,
        attack_fn=attack_fn,
        verbose=True,
        evaluate_test=True,
    )

    print("\n[4/4] Saving artifacts...")
    saved = save_defense_artifacts(
        result,
        baseline_val,
        baseline_test,
        results_dir=args.results_dir,
    )

    # Summary comparison
    def_acc = result.final_test_metrics.get("accuracy", float("nan"))
    def_f1 = result.final_test_metrics.get("f1", float("nan"))
    def_auc = result.final_test_metrics.get("roc_auc", float("nan"))
    base_acc = baseline_test.get("accuracy", float("nan"))

    print("\n" + "=" * 68)
    print("TASK 3 DEFENSE RESULTS (TEST SET)")
    print("=" * 68)
    print(f"Metric       | Baseline FedAvg | With Defence | Improvement")
    print(f"------------ | --------------- | ------------ | -----------")
    print(
        f"Accuracy     | {base_acc:15.4f} | {def_acc:12.4f} | "
        f"{def_acc - base_acc:+10.4f}"
        if not (np.isnan(base_acc) or np.isnan(def_acc))
        else f"Accuracy     |            n/a | {def_acc:12.4f} |        n/a"
    )
    print(f"F1 Score     |             n/a | {def_f1:12.4f} |        n/a")
    print(f"ROC-AUC      |             n/a | {def_auc:12.4f} |        n/a")
    avg_excl = np.mean(result.n_excluded_per_round) if result.n_excluded_per_round else 0
    print(f"Avg excluded | {'n/a':15s} | {avg_excl:12.1f} |        n/a")
    print("-" * 68)
    print(f"Artifacts saved under: {args.results_dir.resolve()}")
    for key, path in saved.items():
        print(f"  - {key}: {path.name}")
    print("=" * 68 + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
