"""Tests for federated learning module (Task 2).

Verifies:
- weighted-average correctness
- client partition covers all rows and preserves stratification
- identical models average to themselves
- same seed reproduces the same result
- tolerance acceptance criteria between FedAvg and centralized model
- input validation and edge cases
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.config import Config
from src.data import load_data, prepare_data
from src.evaluate import evaluate_model
from src.federated import (
    create_model_from_params,
    extract_params,
    fedavg,
    partition_dataset,
    predict_from_params,
    predict_proba_from_params,
    run_federated,
    train_centralized_model,
    train_local_logistic,
    train_local_step,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Required deliverable tests
# ---------------------------------------------------------------------------

def test_weighted_average_correctness():
    """Verify that fedavg correctly computes mathematically exact weighted averages."""
    updates = [
        {"coef": np.array([1.0, 2.0, 3.0]), "intercept": 10.0},
        {"coef": np.array([5.0, 6.0, 7.0]), "intercept": 20.0},
        {"coef": np.array([9.0, 10.0, 11.0]), "intercept": 30.0},
    ]

    # Test 1: Equal weights (weights=None) -> arithmetic mean
    res_uniform = fedavg(updates)
    expected_coef_uniform = np.array([5.0, 6.0, 7.0])
    expected_intercept_uniform = 20.0
    np.testing.assert_allclose(res_uniform["coef"], expected_coef_uniform)
    assert pytest.approx(expected_intercept_uniform) == res_uniform["intercept"]

    # Test 2: Unequal weights [1, 2, 1]
    # sum = 4. weights = [0.25, 0.5, 0.25]
    # coef = 0.25*[1,2,3] + 0.5*[5,6,7] + 0.25*[9,10,11] = [0.25+2.5+2.25, ...] = [5.0, 6.0, 7.0]
    weights = [1.0, 2.0, 1.0]
    res_weighted = fedavg(updates, weights)
    np.testing.assert_allclose(res_weighted["coef"], np.array([5.0, 6.0, 7.0]))
    assert pytest.approx(20.0) == res_weighted["intercept"]

    # Test 3: Asymmetric weights [1.0, 3.0, 0.0]
    # sum = 4.
    # coef = (1 * [1,2,3] + 3 * [5,6,7]) / 4 = [16/4, 20/4, 24/4] = [4.0, 5.0, 6.0]
    # intercept = (1 * 10 + 3 * 20) / 4 = 70 / 4 = 17.5
    weights_asym = [1.0, 3.0, 0.0]
    res_asym = fedavg(updates, weights_asym)
    np.testing.assert_allclose(res_asym["coef"], np.array([4.0, 5.0, 6.0]))
    assert pytest.approx(17.5) == res_asym["intercept"]


def test_client_partition_covers_all_rows(synthetic_framingham):
    """Verify client partition covers all rows, has no overlaps, and preserves all data."""
    X = synthetic_framingham.drop(columns=["id", "TenYearCHD"])
    y = synthetic_framingham["TenYearCHD"]

    for n_clients in (2, 5, 10, 20):
        partitions = partition_dataset(X, y, n_clients=n_clients, seed=42, stratified=True)

        assert len(partitions) == n_clients

        all_indices: set[int] = set()
        total_rows = 0

        for client in partitions:
            client_idx = set(client.X.index)
            # Must be disjoint with previous partitions
            assert client_idx.isdisjoint(all_indices), f"Overlap detected in client {client.client_id}"
            all_indices.update(client_idx)
            total_rows += client.n_samples

            # Aligned X and y
            assert len(client.X) == len(client.y)
            assert client.X.index.equals(client.y.index)

            # Both classes represented in each shard when stratified
            assert client.positive_count > 0, f"Client {client.client_id} has 0 positive samples"
            assert client.negative_count > 0, f"Client {client.client_id} has 0 negative samples"

        # Complete coverage
        assert all_indices == set(X.index)
        assert total_rows == len(X)


def test_identical_models_average_to_themselves():
    """Verify that identical model updates average to themselves regardless of weights."""
    target_coef = np.array([-0.35, 1.42, 0.08, -2.15, 0.99])
    target_intercept = 0.4285

    for n_clients in (2, 5, 20):
        updates = [
            {"coef": target_coef.copy(), "intercept": target_intercept}
            for _ in range(n_clients)
        ]

        # Uniform weights
        res_uniform = fedavg(updates)
        np.testing.assert_allclose(res_uniform["coef"], target_coef, atol=1e-12)
        assert pytest.approx(target_intercept, abs=1e-12) == res_uniform["intercept"]

        # Random positive weights
        rng = np.random.default_rng(123 + n_clients)
        weights = rng.uniform(1.0, 100.0, size=n_clients)
        res_weighted = fedavg(updates, weights)
        np.testing.assert_allclose(res_weighted["coef"], target_coef, atol=1e-12)
        assert pytest.approx(target_intercept, abs=1e-12) == res_weighted["intercept"]


def test_same_seed_reproduces_the_same_result(synthetic_framingham):
    """Verify that runs with the identical seed produce bit-for-bit reproducible results."""
    bundle = prepare_data(synthetic_framingham, random_state=42)
    config = Config(seed=42, n_clients=4, epochs=5, lr=0.2)

    run_1 = run_federated(bundle, config=config, protocol="fedavg", verbose=False)
    run_2 = run_federated(bundle, config=config, protocol="fedavg", verbose=False)

    np.testing.assert_array_equal(run_1.global_params["coef"], run_2.global_params["coef"])
    assert run_1.global_params["intercept"] == run_2.global_params["intercept"]
    assert run_1.final_val_metrics["accuracy"] == run_2.final_val_metrics["accuracy"]
    assert run_1.final_val_metrics["f1"] == run_2.final_val_metrics["f1"]
    assert run_1.final_test_metrics["accuracy"] == run_2.final_test_metrics["accuracy"]

    # History per epoch must match exactly
    for ep1, ep2 in zip(run_1.history, run_2.history):
        assert ep1["epoch"] == ep2["epoch"]
        assert ep1["accuracy"] == ep2["accuracy"]
        assert ep1["f1"] == ep2["f1"]
        assert ep1["roc_auc"] == ep2["roc_auc"]


# ---------------------------------------------------------------------------
# Robustness, validation, and acceptance criteria tests
# ---------------------------------------------------------------------------

def test_different_seed_produces_different_partitions(synthetic_framingham):
    """Verify that different seeds produce different client assignments."""
    X = synthetic_framingham.drop(columns=["id", "TenYearCHD"])
    y = synthetic_framingham["TenYearCHD"]

    p1 = partition_dataset(X, y, n_clients=5, seed=42)
    p2 = partition_dataset(X, y, n_clients=5, seed=99)

    assert not p1[0].X.index.equals(p2[0].X.index)


def test_fedavg_input_validation():
    """Verify error handling on invalid inputs to fedavg."""
    with pytest.raises(ValueError, match="updates must not be empty"):
        fedavg([])

    with pytest.raises(ValueError, match="weights must have shape"):
        fedavg([{"coef": np.zeros(2), "intercept": 0.0}], weights=[1.0, 2.0])

    with pytest.raises(ValueError, match="All weights must be non-negative"):
        fedavg(
            [{"coef": np.zeros(2), "intercept": 0.0}, {"coef": np.zeros(2), "intercept": 0.0}],
            weights=[1.0, -0.5],
        )

    with pytest.raises(ValueError, match="Sum of weights must be strictly positive"):
        fedavg(
            [{"coef": np.zeros(2), "intercept": 0.0}],
            weights=[0.0],
        )

    with pytest.raises(ValueError, match="Weights must not contain NaN"):
        fedavg(
            [{"coef": np.zeros(2), "intercept": 0.0}],
            weights=[np.nan],
        )

    with pytest.raises(ValueError, match="coef shape"):
        fedavg([
            {"coef": np.zeros(2), "intercept": 0.0},
            {"coef": np.zeros(3), "intercept": 0.0},
        ])

    with pytest.raises(KeyError):
        fedavg([{"coef": np.zeros(2)}])


def test_create_model_from_params_and_vectorized_predict(synthetic_framingham):
    """Verify model reconstruction from params produces correct predictions and probabilities."""
    bundle = prepare_data(synthetic_framingham, random_state=42)
    clf = LogisticRegression(class_weight="balanced", random_state=42)
    clf.fit(bundle.X_train, bundle.y_train)

    params = extract_params(clf)
    assert params["coef"].shape == (bundle.n_features,)
    assert isinstance(params["intercept"], float)

    reconstructed_model = create_model_from_params(params, bundle.feature_names)
    assert reconstructed_model.classes_.tolist() == [0, 1]
    assert reconstructed_model.n_features_in_ == bundle.n_features

    # Check predictions match original fitted model
    pred_orig = clf.predict(bundle.X_val)
    pred_recon = reconstructed_model.predict(bundle.X_val)
    np.testing.assert_array_equal(pred_orig, pred_recon)

    # Check vectorized functions match
    probs_vec = predict_proba_from_params(bundle.X_val, params)
    probs_orig = clf.predict_proba(bundle.X_val)
    np.testing.assert_allclose(probs_vec, probs_orig, atol=1e-5)

    preds_vec = predict_from_params(bundle.X_val, params)
    np.testing.assert_array_equal(preds_vec, pred_orig)


def test_train_centralized_model_returns_valid_metrics(synthetic_framingham):
    """Verify centralized model training returns complete metrics."""
    bundle = prepare_data(synthetic_framingham, random_state=42)
    model, val_m, test_m = train_centralized_model(bundle, random_state=42)

    for metrics, name in [(val_m, "val"), (test_m, "test")]:
        assert 0.0 <= metrics["accuracy"] <= 1.0, f"{name} accuracy out of range"
        assert 0.0 <= metrics["precision"] <= 1.0, f"{name} precision out of range"
        assert 0.0 <= metrics["recall"] <= 1.0, f"{name} recall out of range"
        assert 0.0 <= metrics["f1"] <= 1.0, f"{name} f1 out of range"
        assert 0.0 <= metrics["roc_auc"] <= 1.0, f"{name} roc_auc out of range"
        assert "confusion_matrix" in metrics
        assert len(metrics["confusion_matrix"]) == 2


def test_fedavg_is_within_tolerance_of_centralized(synthetic_framingham):
    """Verify acceptance criteria: FedAvg is within +/- 3% accuracy of centralized model."""
    bundle = prepare_data(synthetic_framingham, random_state=42)
    config = Config(seed=42, n_clients=5, epochs=30, lr=0.2)

    # 1. Centralized reference
    _, _, cent_test = train_centralized_model(bundle, random_state=42)

    # 2. FedAvg baseline
    fed_res = run_federated(bundle, config=config, protocol="fedavg", verbose=False)

    cent_acc = cent_test["accuracy"]
    fed_acc = fed_res.final_test_metrics["accuracy"]
    diff = abs(fed_acc - cent_acc)

    # Suggested tolerance is within +/- 2-3%
    assert diff <= 0.03, f"Accuracy difference {diff:.4f} exceeds 3.0% (Cent: {cent_acc:.4f}, Fed: {fed_acc:.4f})"


def test_federated_cli_end_to_end(synthetic_csv, tmp_path):
    """Verify that the baseline run command runs end-to-end and saves all artifacts."""
    results_dir = tmp_path / "baseline"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.federated",
            "--data",
            str(synthetic_csv),
            "--results-dir",
            str(results_dir),
            "--epochs",
            "10",
            "--n-clients",
            "4",
            "--seed",
            "42",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, f"CLI failed: {result.stderr}\nStdout: {result.stdout}"
    assert "BASELINE COMPARISON: CENTRALIZED vs FEDAVG" in result.stdout
    assert "Acceptance Criteria Tolerance Check: PASSED" in result.stdout

    # Check all 10 expected artifact files
    assert (results_dir / "fedavg_val_metrics.json").exists()
    assert (results_dir / "fedavg_test_metrics.json").exists()
    assert (results_dir / "fedavg_test_confusion_matrix.png").exists()
    assert (results_dir / "centralized_val_metrics.json").exists()
    assert (results_dir / "centralized_test_metrics.json").exists()
    assert (results_dir / "centralized_test_confusion_matrix.png").exists()
    assert (results_dir / "convergence.csv").exists()
    assert (results_dir / "convergence.png").exists()
    assert (results_dir / "comparison.json").exists()
    assert (results_dir / "baseline_config.json").exists()

    # Validate contents of comparison.json
    with (results_dir / "comparison.json").open("r", encoding="utf-8") as f:
        comp = json.load(f)
    assert comp["tolerance_satisfied"] is True
    assert "test_comparison" in comp
