#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Leakage-safe static-FC logistic baseline on the CiMyGn outer splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from cimygn_data import (
    Subject,
    safe_subject_standardize,
    scan_dataset,
    site_class_strata,
    split_fit_validation,
)
from cimygn_engine import (
    METRIC_NAMES,
    metrics_from_predictions,
    safe_auc,
    select_balanced_threshold,
    stratified_bootstrap_intervals,
)


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def static_fc_features(
    subjects: Sequence[Subject], standardization_eps: float
) -> np.ndarray:
    n_rois = subjects[0].x.shape[1]
    row, column = np.triu_indices(n_rois, k=1)
    features = np.empty((len(subjects), len(row)), dtype=np.float32)
    for index, subject in enumerate(subjects):
        x, _ = safe_subject_standardize(subject.x, standardization_eps)
        value = np.asarray(x, dtype=np.float64)
        value -= value.mean(axis=0, keepdims=True)
        gram = value.T @ value
        norm = np.sqrt(np.maximum(np.diag(gram), 0.0))
        denominator = norm[:, None] * norm[None, :]
        correlation = np.zeros_like(gram)
        np.divide(gram, denominator, out=correlation, where=denominator > 1e-12)
        edges = np.clip(correlation[row, column], -0.999, 0.999)
        features[index] = np.arctanh(edges).astype(np.float32)
    if not np.all(np.isfinite(features)):
        raise FloatingPointError("Static-FC features contain NaN/Inf.")
    return features


def sample_weights(subjects: Sequence[Subject], strategy: str) -> np.ndarray:
    if strategy == "uniform":
        return np.ones(len(subjects), dtype=np.float64)
    if strategy == "class":
        keys = [str(subject.label) for subject in subjects]
    elif strategy == "site-class":
        keys = [f"S{subject.site}_Y{subject.label}" for subject in subjects]
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    counts = Counter(keys)
    weight = np.asarray([1.0 / counts[key] for key in keys], dtype=np.float64)
    return weight / weight.mean()


def fit_transformer(
    x_fit: np.ndarray,
    x_validation: np.ndarray,
    x_test: np.ndarray,
    pca_components: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    scaler = StandardScaler()
    fit_scaled = scaler.fit_transform(x_fit)
    validation_scaled = scaler.transform(x_validation)
    test_scaled = scaler.transform(x_test)
    n_components = min(pca_components, len(x_fit) - 1, x_fit.shape[1])
    if n_components < 1:
        raise ValueError("The requested split is too small for PCA.")
    solver = "randomized" if n_components < min(fit_scaled.shape) else "full"
    pca = PCA(n_components=n_components, svd_solver=solver, random_state=seed)
    fit_reduced = pca.fit_transform(fit_scaled)
    validation_reduced = pca.transform(validation_scaled)
    test_reduced = pca.transform(test_scaled)
    diagnostics = {
        "n_input_edges": int(x_fit.shape[1]),
        "n_pca_components": int(n_components),
        "pca_explained_variance_ratio_sum": float(
            pca.explained_variance_ratio_.sum()
        ),
        "scaler_fit_subjects": int(len(x_fit)),
    }
    return fit_reduced, validation_reduced, test_reduced, diagnostics


def train_candidate(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    weights: np.ndarray,
    c_value: float,
    seed: int,
) -> LogisticRegression:
    model = LogisticRegression(
        C=c_value,
        solver="liblinear",
        max_iter=5000,
        random_state=seed,
    )
    model.fit(x_fit, y_fit, sample_weight=weights)
    return model


def run_split(
    subjects: Sequence[Subject],
    features: np.ndarray,
    outer_train_index: np.ndarray,
    test_index: np.ndarray,
    protocol: str,
    split_name: str,
    args,
) -> Dict:
    split_dir = args.outdir / protocol / split_name
    split_seed = args.seed + int(
        hashlib.sha256(f"{protocol}/{split_name}".encode("utf-8")).hexdigest()[:8], 16
    ) % 1_000_000
    outer_train_subjects = [subjects[int(i)] for i in outer_train_index]
    fit_relative, validation_relative, validation_scheme = split_fit_validation(
        outer_train_subjects, args.validation_fraction, split_seed
    )
    fit_index = outer_train_index[fit_relative]
    validation_index = outer_train_index[validation_relative]
    fit_subjects = [subjects[int(i)] for i in fit_index]

    metadata_path = split_dir / "metadata.json"
    if metadata_path.is_file() and args.resume:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("status") == "complete"
            and metadata.get("experiment_signature") == args.experiment_signature
        ):
            print(f"[RESUME] Reusing static-FC {protocol}/{split_name}")
            return {"Protocol": protocol, "Split": split_name, **metadata["metrics"]}
        raise RuntimeError(
            f"Incompatible output exists at {split_dir}; use a new --outdir."
        )
    if split_dir.exists() and any(split_dir.iterdir()):
        if args.resume:
            print(f"[RERUN] Removing incomplete static-FC output: {split_dir}")
            shutil.rmtree(split_dir)
        else:
            raise RuntimeError(
                f"Incomplete output exists at {split_dir}; enable --resume or use a new --outdir."
            )
    split_dir.mkdir(parents=True, exist_ok=True)

    y = np.asarray([subject.label for subject in subjects], dtype=int)
    x_fit, x_validation, x_test, diagnostics = fit_transformer(
        features[fit_index],
        features[validation_index],
        features[test_index],
        args.pca_components,
        split_seed,
    )
    y_fit = y[fit_index]
    y_validation = y[validation_index]
    y_test = y[test_index]
    weights = sample_weights(fit_subjects, args.sampling_strategy)

    candidate_rows = []
    candidates = []
    for c_value in sorted(set(args.c_grid)):
        model = train_candidate(
            x_fit, y_fit, weights, float(c_value), split_seed
        )
        probability = model.predict_proba(x_validation)[:, 1]
        auc = safe_auc(y_validation, probability)
        threshold = (
            0.5
            if args.threshold_strategy == "fixed-0.5"
            else select_balanced_threshold(y_validation, probability)
        )
        balanced_accuracy = float(
            balanced_accuracy_score(
                y_validation, (probability >= threshold).astype(int)
            )
        )
        candidates.append((auc, balanced_accuracy, -float(c_value), model, threshold))
        candidate_rows.append(
            {
                "C": float(c_value),
                "validation_ROC_AUC": auc,
                "validation_Balanced_Accuracy": balanced_accuracy,
                "validation_threshold": threshold,
                "validation_probability_sd": float(probability.std(ddof=0)),
            }
        )
    _, _, _, selected_model, threshold = max(
        candidates, key=lambda item: item[:3]
    )
    selected_c = float(selected_model.C)
    validation_probability = selected_model.predict_proba(x_validation)[:, 1]
    test_probability = selected_model.predict_proba(x_test)[:, 1]
    test_prediction = (test_probability >= threshold).astype(int)
    metrics = metrics_from_predictions(y_test, test_probability, test_prediction)

    pd.DataFrame(candidate_rows).to_csv(
        split_dir / "validation_model_selection.csv", index=False
    )
    pd.DataFrame(
        {
            "subject_id": [subjects[int(i)].subject_id for i in validation_index],
            "site": [subjects[int(i)].site for i in validation_index],
            "y_true": y_validation,
            "prob_MDD": validation_probability,
            "threshold": threshold,
            "y_pred": (validation_probability >= threshold).astype(int),
        }
    ).to_csv(split_dir / "validation_predictions.csv", index=False)
    prediction_table = pd.DataFrame(
        {
            "Protocol": protocol,
            "Split": split_name,
            "subject_id": [subjects[int(i)].subject_id for i in test_index],
            "site": [subjects[int(i)].site for i in test_index],
            "group": [subjects[int(i)].group for i in test_index],
            "path": [str(subjects[int(i)].path) for i in test_index],
            "y_true": y_test,
            "prob_MDD": test_probability,
            "threshold": threshold,
            "y_pred": test_prediction,
        }
    )
    prediction_table.to_csv(split_dir / "predictions.csv", index=False)
    pd.DataFrame(
        [
            {
                "role": role,
                "subject_id": subjects[int(index)].subject_id,
                "site": subjects[int(index)].site,
                "group": subjects[int(index)].group,
                "path": str(subjects[int(index)].path),
            }
            for role, indices in (
                ("fit", fit_index),
                ("validation", validation_index),
                ("test", test_index),
            )
            for index in indices
        ]
    ).to_csv(split_dir / "split_manifest.csv", index=False)
    write_json(
        metadata_path,
        {
            "status": "complete",
            "experiment_signature": args.experiment_signature,
            "protocol": protocol,
            "split": split_name,
            "split_seed": split_seed,
            "validation_scheme": validation_scheme,
            "n_fit": int(len(fit_index)),
            "n_validation": int(len(validation_index)),
            "n_test": int(len(test_index)),
            "selected_C": selected_c,
            "selected_threshold": float(threshold),
            "preprocessing": diagnostics,
            "metrics": metrics,
        },
    )
    print(
        f"[STATIC-FC] {protocol}/{split_name}: C={selected_c:g}, "
        f"BA={metrics['Balanced-Accuracy']:.4f}, AUC={metrics['ROC-AUC']:.4f}, "
        f"PR-AUC={metrics['PR-AUC']:.4f}"
    )
    return {"Protocol": protocol, "Split": split_name, **metrics}


def summarize(args) -> None:
    metric_rows = []
    prediction_tables = []
    for protocol in ("kfold", "loso"):
        protocol_dir = args.outdir / protocol
        if not protocol_dir.is_dir():
            continue
        for metadata_path in sorted(protocol_dir.glob("*/metadata.json")):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                metadata.get("status") != "complete"
                or metadata.get("experiment_signature") != args.experiment_signature
            ):
                continue
            metric_rows.append(
                {
                    "Protocol": protocol,
                    "Split": metadata["split"],
                    **metadata["metrics"],
                }
            )
            prediction_tables.append(pd.read_csv(metadata_path.parent / "predictions.csv"))
    if not metric_rows:
        return
    metrics = pd.DataFrame(metric_rows).sort_values(["Protocol", "Split"])
    predictions = pd.concat(prediction_tables, ignore_index=True)
    metrics.to_csv(args.outdir / "static_fc_per_split_metrics.csv", index=False)
    macro_rows = []
    pooled_rows = []
    for protocol in ("kfold", "loso"):
        split_metrics = metrics[metrics["Protocol"] == protocol]
        pooled = predictions[predictions["Protocol"] == protocol]
        if split_metrics.empty:
            continue
        macro = {"Protocol": protocol, "n_splits": int(len(split_metrics))}
        for metric in METRIC_NAMES:
            macro[f"{metric}_mean"] = float(split_metrics[metric].mean())
            macro[f"{metric}_sd"] = (
                float(split_metrics[metric].std(ddof=1))
                if len(split_metrics) > 1
                else 0.0
            )
        macro_rows.append(macro)
        pooled_metrics = metrics_from_predictions(
            pooled["y_true"].to_numpy(dtype=int),
            pooled["prob_MDD"].to_numpy(dtype=float),
            pooled["y_pred"].to_numpy(dtype=int),
        )
        intervals = stratified_bootstrap_intervals(
            pooled,
            args.bootstrap_iterations,
            args.seed + (17 if protocol == "kfold" else 29),
        )
        pooled_rows.append(
            {
                "Protocol": protocol,
                "n_completed_splits": int(len(split_metrics)),
                "n_pooled_subjects": int(len(pooled)),
                **pooled_metrics,
                **intervals,
            }
        )
        print(
            f"[STATIC-FC SUMMARY] {protocol}: n={len(pooled)}, "
            f"BA={pooled_metrics['Balanced-Accuracy']:.4f}, "
            f"AUC={pooled_metrics['ROC-AUC']:.4f}, "
            f"PR-AUC={pooled_metrics['PR-AUC']:.4f} "
            f"(prevalence={pooled_metrics['MDD-Prevalence']:.4f})"
        )
    pd.DataFrame(macro_rows).to_csv(
        args.outdir / "static_fc_summary_macro.csv", index=False
    )
    pd.DataFrame(pooled_rows).to_csv(
        args.outdir / "static_fc_summary_pooled.csv", index=False
    )
    predictions.to_csv(args.outdir / "static_fc_predictions_pooled.csv", index=False)


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Static-FC baseline using the same K-fold/LOSO design as CiMyGn.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            r"C:\z77HAO\SMBU\HMM4analysis\Datasets_3\REST-Meta_fMRI_MDD_7sites"
        ),
    )
    parser.add_argument("--sites", type=int, nargs="+", default=[1, 15, 20, 21, 25])
    parser.add_argument("--protocol", choices=["both", "kfold", "loso"], default="both")
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--cv-fold", type=int, default=None)
    parser.add_argument("--loso-site", type=int, default=None)
    parser.add_argument("--outdir", type=Path, default=Path("results_static_fc_baseline"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-rois", type=int, default=90)
    parser.add_argument("--timepoints", type=int, default=200)
    parser.add_argument("--standardization-eps", type=float, default=1e-8)
    parser.add_argument("--max-constant-rois", type=int, default=20)
    parser.add_argument("--strict-data", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--pca-components", type=int, default=100)
    parser.add_argument("--c-grid", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0])
    parser.add_argument(
        "--sampling-strategy",
        choices=["site-class", "class", "uniform"],
        default="site-class",
    )
    parser.add_argument(
        "--threshold-strategy",
        choices=["validation-balanced", "fixed-0.5"],
        default="validation-balanced",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args(argv)


def validate_args(args):
    args.data_root = args.data_root.expanduser().resolve()
    args.outdir = args.outdir.expanduser().resolve()
    args.sites = list(dict.fromkeys(args.sites))
    if len(args.sites) < 2 or args.n_splits < 2:
        raise ValueError("At least two sites and two folds are required.")
    if args.cv_fold is not None and not 1 <= args.cv_fold <= args.n_splits:
        raise ValueError("--cv-fold is outside the valid range.")
    if args.loso_site is not None and args.loso_site not in args.sites:
        raise ValueError("--loso-site must also be present in --sites.")
    if args.pca_components < 1 or any(value <= 0 for value in args.c_grid):
        raise ValueError("PCA components and all C values must be positive.")
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be in (0,0.5).")
    if args.bootstrap_iterations < 0:
        raise ValueError("--bootstrap-iterations cannot be negative.")
    args.outdir.mkdir(parents=True, exist_ok=True)
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = validate_args(parse_args(argv))
    print("Data root:", args.data_root)
    print("Output:", args.outdir)
    subjects = scan_dataset(args)
    source_directory = Path(__file__).resolve().parent
    source_hashes = {
        name: hashlib.sha256((source_directory / name).read_bytes()).hexdigest()
        for name in (
            "run_static_fc_baseline.py",
            "cimygn_data.py",
            "cimygn_engine.py",
        )
    }
    signature_payload = {
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key not in {"cv_fold", "loso_site", "check_only", "resume"}
        },
        "source_sha256": source_hashes,
        "subjects": [
            (subject.subject_id, subject.content_sha256) for subject in subjects
        ],
    }
    args.experiment_signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    write_json(
        args.outdir / "run_config.json",
        {
            "experiment_signature": args.experiment_signature,
            "signature_payload": signature_payload,
        },
    )
    if args.check_only:
        print("Static-FC dataset preflight completed; no model was fitted.")
        return 0

    print("Computing per-subject Fisher-z FC features...")
    features = static_fc_features(subjects, args.standardization_eps)
    labels = np.asarray([subject.label for subject in subjects], dtype=int)
    sites = np.asarray([subject.site for subject in subjects], dtype=int)
    indices = np.arange(len(subjects))
    if args.protocol in ("both", "kfold"):
        strata = site_class_strata(subjects)
        counts = pd.Series(strata).value_counts()
        if int(counts.min()) < args.n_splits:
            raise RuntimeError("The smallest site-by-class cell cannot support K-fold CV.")
        splitter = StratifiedKFold(
            n_splits=args.n_splits, shuffle=True, random_state=args.seed
        )
        for fold, (train_index, test_index) in enumerate(
            splitter.split(indices, strata), start=1
        ):
            if args.cv_fold is not None and fold != args.cv_fold:
                continue
            run_split(
                subjects,
                features,
                train_index,
                test_index,
                "kfold",
                f"fold_{fold:02d}",
                args,
            )
            summarize(args)
    if args.protocol in ("both", "loso"):
        for held_site in args.sites:
            if args.loso_site is not None and held_site != args.loso_site:
                continue
            test_index = indices[sites == held_site]
            train_index = indices[sites != held_site]
            if len(np.unique(labels[test_index])) != 2:
                raise RuntimeError(f"Held-out site {held_site} lacks one class.")
            run_split(
                subjects,
                features,
                train_index,
                test_index,
                "loso",
                f"site_{held_site}",
                args,
            )
            summarize(args)
    summarize(args)
    print("Completed. Static-FC results:", args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
