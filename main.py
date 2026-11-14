"""FedDef command-line entry point.

Task 1 (foundation) responsibilities:
- load and validate the Framingham dataset
- create the shared, reproducible train/validation/test split
- save ``results/data_summary.json`` and ``results/run_config.json``
- optionally run a centralized logistic-regression smoke check

FedAvg, attacks, and the defense are implemented in their own modules by
Tasks 2-4.  The ``--smoke-train`` flag is only a data/evaluation sanity check,
not the project's FedAvg baseline.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from sklearn.linear_model import LogisticRegression

from src.config import Config
from src.data import DataBundle, load_data, save_data_summary, save_splits
from src.evaluate import (
    evaluate_model,
    format_metrics,
    save_confusion_matrix_plot,
    save_metrics,
)
from src.federated import (
    run_federated,
    save_baseline_artifacts,
    train_centralized_model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FedDef data foundation and sanity check.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help=(
            "Path to framingham.csv. Defaults to data/framingham.csv, "
            "then framingham.csv in the repository root."
        ),
    )
    parser.add_argument(
        "--target",
        default="TenYearCHD",
        help="Name of the binary target column (default: TenYearCHD).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the split and smoke model (default: 42).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of rows reserved for the test set (default: 0.2).",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Fraction of all rows reserved for validation (default: 0.1).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory for generated summaries and metrics (default: results).",
    )
    parser.add_argument(
        "--save-splits",
        action="store_true",
        help="Also save the prepared train/val/test CSV files.",
    )
    parser.add_argument(
        "--smoke-train",
        action="store_true",
        help=(
            "Fit a temporary centralized logistic regression and evaluate it "
            "on the validation set. This is a pipeline sanity check, not the "
            "Task 2 FedAvg baseline."
        ),
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run Task 2 clean FedAvg baseline and compare with centralized model.",
    )
    parser.add_argument(
        "--defense",
        action="store_true",
        help="Run Task 3 FEDDEF robust aggregation defense against model poisoning.",
    )
    parser.add_argument(
        "--attack-type",
        choices=["random", "sign_flip", "scaling", "label_flip", "collusion", "none"],
        default="random",
        help="Poisoning attack type for --defense (default: random).",
    )
    parser.add_argument(
        "--mal-fraction",
        type=float,
        default=0.2,
        help="Fraction of malicious clients for --defense (default: 0.2).",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=0.25,
        help="Exclusion threshold for FEDDEF anomaly scores (default: 0.25).",
    )
    parser.add_argument(
        "--z0",
        type=float,
        default=1.0,
        help="Z-score threshold for anomaly accumulation (default: 1.0).",
    )
    parser.add_argument(
        "--protocol",
        choices=["fedavg", "one_shot"],
        default="fedavg",
        help="FL protocol for --baseline (default: fedavg).",
    )
    parser.add_argument(
        "--n-clients",
        type=int,
        default=20,
        help="Number of clients for --baseline (default: 20).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Communication rounds for --baseline (default: 50).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.2,
        help="Client learning rate for --baseline (default: 0.2).",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help=(
            "With --smoke-train, also evaluate on the held-out test set. "
            "Use this only for a final check, not during tuning."
        ),
    )
    return parser


def resolve_data_path(cli_path: Path | None) -> Path:
    """Resolve the dataset path without silently guessing wrong files."""
    if cli_path is not None:
        return cli_path

    for candidate in (Path("data/framingham.csv"), Path("framingham.csv")):
        if candidate.exists():
            return candidate
    return Path("data/framingham.csv")


def print_bundle_summary(bundle: DataBundle) -> None:
    metadata = bundle.metadata
    print("=" * 68)
    print("FedDef - Task 1 data foundation")
    print("=" * 68)
    print(f"Source:          {metadata.get('source') or 'unknown'}")
    print(
        "Rows:            "
        f"{metadata['n_rows_after_target_drop']:,} used / "
        f"{metadata['n_rows_original']:,} original"
    )
    print(f"Target:          {bundle.target_name}")
    print(f"Features:        {bundle.n_features}")
    print(
        "Dropped:         "
        f"{metadata['dropped_columns'] if metadata['dropped_columns'] else 'none'}"
    )
    print(
        "Missing values:  "
        f"{sum(metadata['missing_values_before_imputation'].values()):,} imputed "
        "from training statistics"
    )
    print()
    for split_name in ("train", "val", "test"):
        size = metadata["split_sizes"][split_name]
        positive_rate = metadata["positive_rate"][split_name]
        print(f"{split_name.capitalize():>5}: {size:>5,} rows | {positive_rate:6.2%} positive")
    print()
    print("Feature names:")
    for name in bundle.feature_names:
        print(f"  - {name}")


def run_smoke_check(
    bundle: DataBundle,
    results_dir: Path,
    seed: int,
    evaluate_test: bool,
) -> None:
    """Fit a centralized baseline as a data/evaluation sanity check."""
    print()
    print("-" * 68)
    print("Smoke check: centralized LogisticRegression (validation set)")
    print("-" * 68)

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(bundle.X_train, bundle.y_train)

    val_metrics = evaluate_model(model, bundle.X_val, bundle.y_val)
    save_metrics(val_metrics, results_dir / "smoke_val_metrics.json")
    save_confusion_matrix_plot(
        val_metrics,
        results_dir / "smoke_val_confusion_matrix.png",
        title="Validation confusion matrix (smoke check)",
    )
    print(format_metrics(val_metrics))
    print(f"Saved: {results_dir / 'smoke_val_metrics.json'}")

    if evaluate_test:
        test_metrics = evaluate_model(model, bundle.X_test, bundle.y_test)
        save_metrics(test_metrics, results_dir / "smoke_test_metrics.json")
        save_confusion_matrix_plot(
            test_metrics,
            results_dir / "smoke_test_confusion_matrix.png",
            title="Test confusion matrix (final check only)",
        )
        print()
        print("Held-out test metrics (final check):")
        print(format_metrics(test_metrics))
        print(f"Saved: {results_dir / 'smoke_test_metrics.json'}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.evaluate_test and not args.smoke_train:
        parser.error("--evaluate-test requires --smoke-train")

    config = Config(
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
        target_col=args.target,
        data_path=resolve_data_path(args.data),
        results_dir=args.results_dir,
    )
    config.ensure_results_dir()
    config.save(config.results_dir / "run_config.json")

    if not config.data_path.exists():
        print(f"Dataset not found: {config.data_path}", file=sys.stderr)
        print(
            "Place framingham.csv in data/framingham.csv "
            "(see data/README.md) or pass --data <path>.",
            file=sys.stderr,
        )
        return 1

    try:
        bundle = load_data(
            config.data_path,
            target_col=config.target_col,
            test_size=config.test_size,
            val_size=config.val_size,
            random_state=config.seed,
        )
    except (FileNotFoundError, ValueError, TypeError) as exc:
        print(f"Data error: {exc}", file=sys.stderr)
        return 1

    summary_path = save_data_summary(
        bundle,
        config.results_dir / "data_summary.json",
    )
    print_bundle_summary(bundle)
    print(f"\nSaved data summary: {summary_path}")

    if args.save_splits:
        splits_dir = save_splits(bundle, config.results_dir / "splits")
        print(f"Saved prepared splits: {splits_dir}")

    if args.smoke_train:
        run_smoke_check(
            bundle,
            results_dir=config.results_dir,
            seed=config.seed,
            evaluate_test=args.evaluate_test,
        )

    if args.baseline:
        print()
        print("=" * 68)
        print("Task 2: Clean FedAvg baseline & Centralized Reference")
        print("=" * 68)
        baseline_dir = config.results_dir / "baseline"

        print("\nTraining centralized reference model...")
        cent_model, cent_val, cent_test = train_centralized_model(
            bundle,
            max_iter=1000,
            class_weight="balanced",
            random_state=config.seed,
        )
        print("Centralized test metrics:")
        print(format_metrics(cent_test))

        fed_config = Config(
            seed=config.seed,
            test_size=config.test_size,
            val_size=config.val_size,
            target_col=config.target_col,
            n_clients=args.n_clients,
            epochs=args.epochs,
            lr=args.lr,
            data_path=config.data_path,
            results_dir=config.results_dir,
        )

        print(f"\nRunning {args.protocol.upper()} federated server loop...")
        fed_result = run_federated(
            bundle,
            config=fed_config,
            protocol=args.protocol,
            evaluate_test=True,
            verbose=True,
        )
        print("\nFedAvg test metrics:")
        print(format_metrics(fed_result.final_test_metrics))

        saved = save_baseline_artifacts(
            fed_result,
            cent_val,
            cent_test,
            results_dir=baseline_dir,
        )

        cent_acc = cent_test["accuracy"]
        fed_acc = fed_result.final_test_metrics["accuracy"]
        acc_diff = fed_acc - cent_acc
        cent_f1 = cent_test["f1"]
        fed_f1 = fed_result.final_test_metrics["f1"]
        f1_diff = fed_f1 - cent_f1
        cent_auc = cent_test.get("roc_auc")
        fed_auc = fed_result.final_test_metrics.get("roc_auc")
        auc_diff = (fed_auc - cent_auc) if (fed_auc is not None and cent_auc is not None) else None

        print("\n" + "=" * 68)
        print("BASELINE COMPARISON: CENTRALIZED vs FEDAVG (TEST SET)")
        print("=" * 68)
        print(f"Metric        | Centralized | FedAvg      | Difference")
        print(f"------------- | ----------- | ----------- | ----------")
        print(f"Accuracy      | {cent_acc:11.4f} | {fed_acc:11.4f} | {acc_diff:+10.4f}")
        print(f"F1 Score      | {cent_f1:11.4f} | {fed_f1:11.4f} | {f1_diff:+10.4f}")
        auc_cent_str = f"{cent_auc:11.4f}" if cent_auc is not None else "        n/a"
        auc_fed_str = f"{fed_auc:11.4f}" if fed_auc is not None else "        n/a"
        auc_diff_str = f"{auc_diff:+10.4f}" if auc_diff is not None else "       n/a"
        print(f"ROC-AUC       | {auc_cent_str} | {auc_fed_str} | {auc_diff_str}")
        print("-" * 68)

        within_tolerance = abs(acc_diff) <= 0.03
        status_str = "PASSED (within +/- 3.0% tolerance)" if within_tolerance else "FAILED (exceeds tolerance)"
        print(f"Acceptance Criteria Tolerance Check: {status_str}")
        print(f"Saved baseline artifacts under: {baseline_dir}")
        for k, p in saved.items():
            print(f"  - {k}: {p.name}")

    if args.defense:
        print()
        print("=" * 68)
        print("Task 3: FEDDEF Robust Aggregation Defense Against Model Poisoning")
        print("=" * 68)
        from src.attacks import make_attack_fn
        from src.defense import run_defense, save_defense_artifacts

        defense_dir = config.results_dir / "defense"
        def_config = Config(
            seed=config.seed,
            test_size=config.test_size,
            val_size=config.val_size,
            target_col=config.target_col,
            n_clients=args.n_clients,
            mal_fraction=args.mal_fraction,
            epochs=args.epochs,
            lr=args.lr,
            theta=args.theta,
            z0=args.z0,
            attack_type=args.attack_type if args.attack_type != "none" else "random",
            data_path=config.data_path,
            results_dir=config.results_dir,
        )

        attack_fn = None if args.attack_type == "none" else make_attack_fn(args.attack_type, strength=1.0)
        print(f"\nRunning FEDDEF defense loop (attack={args.attack_type}, theta={args.theta}, z0={args.z0})...")
        def_result = run_defense(
            bundle,
            config=def_config,
            attack_fn=attack_fn,
            verbose=True,
            evaluate_test=True,
        )
        print("\nDefended test metrics:")
        print(format_metrics(def_result.final_test_metrics))

        baseline_val_metrics = {}
        baseline_test_metrics = {}
        base_val_path = config.results_dir / "baseline" / "fedavg_val_metrics.json"
        base_test_path = config.results_dir / "baseline" / "fedavg_test_metrics.json"
        if base_val_path.exists():
            from src.evaluate import load_metrics
            baseline_val_metrics = load_metrics(base_val_path)
            if base_test_path.exists():
                baseline_test_metrics = load_metrics(base_test_path)

        saved_def = save_defense_artifacts(
            def_result,
            baseline_val_metrics,
            baseline_test_metrics,
            results_dir=defense_dir,
        )
        print(f"\nSaved defense artifacts under: {defense_dir}")
        for k, p in saved_def.items():
            print(f"  - {k}: {p.name}")

    print()
    print("Task 1 foundation complete. FedAvg baseline is Task 2. Defense is Task 3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
