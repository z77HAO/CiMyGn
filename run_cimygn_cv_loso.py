#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""10-fold CV and LOSO evaluation for manuscript-aligned CiMyGn."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from cimygn_data import Subject, fit_full_pca_rotation, scan_dataset, standardized_arrays
from cimygn_engine import (
    METRIC_NAMES,
    SequenceDataset,
    make_loader,
    metrics_from_predictions,
    predict_loader,
    resolve_device,
    train_model,
)


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def model_config(args) -> Dict:
    return {
        "data_dim": args.n_rois,
        "n_power_modes": args.k_power_modes,
        "n_fc_modes": args.q_fc_modes,
        "num_classes": 2,
        "encoder_hidden": args.encoder_hidden,
        "encoder_layers": args.encoder_layers,
        "encoder_dropout": args.encoder_dropout,
        "prior_hidden": args.prior_hidden,
        "classifier_hidden": args.classifier_hidden,
        "classifier_dropout": args.classifier_dropout,
        "tau_power": args.tau_power,
        "tau_fc": args.tau_fc,
        "min_scale": args.min_scale,
        "min_cholesky": args.min_cholesky,
        "covariance_jitter": args.covariance_jitter,
    }


def split_fit_validation(
    outer_train_subjects: Sequence[Subject], validation_fraction: float, seed: int
) -> Tuple[List[Subject], List[Subject]]:
    indices = np.arange(len(outer_train_subjects))
    labels = np.asarray([subject.label for subject in outer_train_subjects], dtype=int)
    fit_idx, val_idx = train_test_split(
        indices,
        test_size=float(validation_fraction),
        stratify=labels,
        random_state=int(seed),
    )
    fit_subjects = [outer_train_subjects[int(i)] for i in fit_idx]
    validation_subjects = [outer_train_subjects[int(i)] for i in val_idx]
    return fit_subjects, validation_subjects


def prepare_pca(
    fit_subjects: Sequence[Subject], args
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    if args.pca:
        return fit_full_pca_rotation(
            standardized_arrays(fit_subjects, args.standardization_eps),
            args.pca_rank_rtol,
        )
    rotation = np.eye(args.n_rois, dtype=np.float32)
    eigenvalues = np.ones(args.n_rois, dtype=np.float64)
    diagnostics = {
        "pca_components": args.n_rois,
        "pca_effective_rank": args.n_rois,
        "pca_orthogonality_max_error": 0.0,
        "pca_enabled": False,
    }
    return rotation, eigenvalues, diagnostics


def save_mode_parameters(model, split_dir: Path) -> None:
    model.eval()
    with torch.no_grad():
        power_scale = model.power_scale_templates().cpu().numpy()
        power_variance = model.power_variance_templates().cpu().numpy()
        fc_templates = model.fc_correlation_templates().cpu().numpy()
    np.savez_compressed(
        split_dir / "mode_parameters.npz",
        regional_scale_templates=power_scale,
        regional_power_templates=power_variance,
        fc_correlation_templates=fc_templates,
    )


def run_split(
    subjects: Sequence[Subject],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    protocol: str,
    split_name: str,
    args,
    device: torch.device,
) -> Dict[str, float]:
    split_dir = args.outdir / protocol / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    outer_train = [subjects[int(i)] for i in train_indices]
    test_subjects = [subjects[int(i)] for i in test_indices]
    split_seed = args.seed + sum(ord(c) for c in f"{protocol}/{split_name}")
    fit_subjects, validation_subjects = split_fit_validation(
        outer_train, args.validation_fraction, split_seed
    )

    print(
        f"\n[{protocol.upper()} | {split_name}] "
        f"fit={len(fit_subjects)}, validation={len(validation_subjects)}, "
        f"test={len(test_subjects)}"
    )

    manifest_rows = []
    for role, collection in (
        ("fit", fit_subjects),
        ("validation", validation_subjects),
        ("test", test_subjects),
    ):
        for subject in collection:
            manifest_rows.append(
                {
                    "role": role,
                    "subject_id": subject.subject_id,
                    "site": subject.site,
                    "group": subject.group,
                    "label": subject.label,
                }
            )
    pd.DataFrame(manifest_rows).to_csv(split_dir / "split_manifest.csv", index=False)

    pca_rotation, eigenvalues, pca_diagnostics = prepare_pca(fit_subjects, args)
    pca_diagnostics["pca_enabled"] = bool(args.pca)
    np.savez_compressed(
        split_dir / "training_only_full_rank_pca.npz",
        rotation=pca_rotation,
        eigenvalues=eigenvalues,
    )
    write_json(split_dir / "preprocessing_diagnostics.json", pca_diagnostics)

    fit_dataset = SequenceDataset(fit_subjects, args.standardization_eps)
    validation_dataset = SequenceDataset(
        validation_subjects, args.standardization_eps
    )
    test_dataset = SequenceDataset(test_subjects, args.standardization_eps)

    trained = train_model(
        model_config(args),
        pca_rotation,
        fit_dataset,
        validation_dataset,
        args,
        device,
        split_seed,
    )
    model = trained["model"]
    pd.DataFrame(trained["history"]).to_csv(
        split_dir / "training_history.csv", index=False
    )
    trained["validation_predictions"].to_csv(
        split_dir / "validation_predictions.csv", index=False
    )
    trained["validation_features"].to_csv(
        split_dir / "validation_dynamic_features.csv", index=False
    )

    test_loader = make_loader(
        test_dataset,
        args.batch_size,
        False,
        args.num_workers,
        split_seed + 2,
        device,
    )
    predictions, dynamic_features = predict_loader(model, test_loader, device)
    predictions.to_csv(split_dir / "predictions.csv", index=False)
    dynamic_features.to_csv(split_dir / "test_dynamic_features.csv", index=False)

    metrics = metrics_from_predictions(
        predictions["y_true"].to_numpy(dtype=int),
        predictions["prob_MDD"].to_numpy(dtype=float),
        predictions["y_pred"].to_numpy(dtype=int),
    )
    save_mode_parameters(model, split_dir)

    if args.save_models:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": model_config(args),
                "pca_rotation": pca_rotation,
                "best_epoch": trained["best_epoch"],
            },
            split_dir / "cimygn_model.pt",
        )

    metadata = {
        "status": "complete",
        "protocol": protocol,
        "split": split_name,
        "n_fit": len(fit_subjects),
        "n_validation": len(validation_subjects),
        "n_test": len(test_subjects),
        "best_epoch": int(trained["best_epoch"]),
        "best_validation_auc": float(trained["best_validation_auc"]),
        "validation_metrics": trained["validation_metrics"],
        "test_metrics": metrics,
        "model_config": model_config(args),
    }
    write_json(split_dir / "metadata.json", metadata)

    print(
        f"[RESULT] {protocol}/{split_name}: "
        f"Acc={metrics['Accuracy']:.4f}, Sen={metrics['Sensitivity']:.4f}, "
        f"Spec={metrics['Specificity']:.4f}, F1={metrics['F1-score']:.4f}, "
        f"AUC={metrics['ROC-AUC']:.4f}"
    )
    return {"Protocol": protocol, "Split": split_name, **metrics}


def kfold_splits(subjects: Sequence[Subject], n_splits: int, seed: int):
    indices = np.arange(len(subjects))
    labels = np.asarray([subject.label for subject in subjects], dtype=int)
    splitter = StratifiedKFold(
        n_splits=int(n_splits), shuffle=True, random_state=int(seed)
    )
    return list(splitter.split(indices, labels))


def loso_splits(subjects: Sequence[Subject], sites: Sequence[int]):
    all_indices = np.arange(len(subjects))
    result = []
    for site in sites:
        test = np.asarray(
            [i for i, subject in enumerate(subjects) if subject.site == site], dtype=int
        )
        train = np.setdiff1d(all_indices, test, assume_unique=True)
        if len(test) == 0:
            raise RuntimeError(f"No subjects found for LOSO site {site}.")
        result.append((site, train, test))
    return result


def summarize(rows: List[Dict[str, float]], outdir: Path) -> None:
    table = pd.DataFrame(rows)
    table.to_csv(outdir / "cimygn_per_split_metrics.csv", index=False)
    summary_rows = []
    for protocol, group in table.groupby("Protocol"):
        row = {"Protocol": protocol, "n_splits": len(group)}
        for metric in METRIC_NAMES:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(
        outdir / "cimygn_summary_macro.csv", index=False
    )


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="CiMyGn 10-fold CV and LOSO evaluation aligned with the manuscript.",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sites", type=int, nargs="+", default=[1, 15, 20, 21, 25])
    parser.add_argument("--protocol", choices=["both", "kfold", "loso"], default="both")
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--outdir", type=Path, default=Path("results_cimygn"))
    parser.add_argument("--seed", type=int, default=42)

    data = parser.add_argument_group("data")
    data.add_argument("--n-rois", type=int, default=90)
    data.add_argument("--timepoints", type=int, default=200)
    data.add_argument("--validation-fraction", type=float, default=0.20)
    data.add_argument("--standardization-eps", type=float, default=1e-8)
    data.add_argument("--pca", action=argparse.BooleanOptionalAction, default=True)
    data.add_argument("--pca-rank-rtol", type=float, default=1e-10)
    data.add_argument("--max-constant-rois", type=int, default=90)
    data.add_argument("--strict-data", action=argparse.BooleanOptionalAction, default=False)

    model = parser.add_argument_group("model")
    model.add_argument("--k-power-modes", type=int, default=3)
    model.add_argument("--q-fc-modes", type=int, default=6)
    model.add_argument("--encoder-hidden", type=int, default=128)
    model.add_argument("--encoder-layers", type=int, default=2)
    model.add_argument("--encoder-dropout", type=float, default=0.2)
    model.add_argument("--prior-hidden", type=int, default=128)
    model.add_argument("--classifier-hidden", type=int, default=256)
    model.add_argument("--classifier-dropout", type=float, default=0.5)
    model.add_argument("--tau-power", type=float, default=1.0)
    model.add_argument("--tau-fc", type=float, default=1.0)
    model.add_argument("--min-scale", type=float, default=1e-4)
    model.add_argument("--min-cholesky", type=float, default=1e-4)
    model.add_argument("--covariance-jitter", type=float, default=1e-5)

    training = parser.add_argument_group("training")
    training.add_argument("--batch-size", type=int, default=32)
    training.add_argument("--epochs", type=int, default=200)
    training.add_argument("--patience", type=int, default=30)
    training.add_argument("--learning-rate", type=float, default=3e-4)
    training.add_argument("--weight-decay", type=float, default=1e-2)
    training.add_argument("--gamma-cls", type=float, default=1.0)
    # Exact KL schedule is not stated in the manuscript. Linear warm-up is explicit.
    training.add_argument("--kl-anneal-epochs", type=int, default=40)
    # Memory chunking only; all T=200 time points remain in the NLL.
    training.add_argument("--nll-chunk-size", type=int, default=16)

    execution = parser.add_argument_group("execution")
    execution.add_argument("--device", default="auto")
    execution.add_argument("--num-workers", type=int, default=0)
    execution.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    execution.add_argument("--cv-fold", type=int, default=None)
    execution.add_argument("--loso-site", type=int, default=None)
    execution.add_argument("--save-models", action="store_true")
    return parser.parse_args(argv)


def validate_args(args) -> None:
    args.data_root = args.data_root.expanduser().resolve()
    args.outdir = args.outdir.expanduser().resolve()
    args.sites = list(dict.fromkeys(args.sites))
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {args.data_root}")
    if args.n_rois != 90:
        print(f"[WARNING] Manuscript uses 90 AAL ROIs; received --n-rois={args.n_rois}.")
    if args.timepoints != 200:
        print(f"[WARNING] Manuscript uses T=200; received --timepoints={args.timepoints}.")
    if args.cv_fold is not None and not 1 <= args.cv_fold <= args.n_splits:
        raise ValueError("--cv-fold must be between 1 and --n-splits.")
    if args.loso_site is not None and args.loso_site not in args.sites:
        raise ValueError("--loso-site must be listed in --sites.")
    if not (0.0 < args.validation_fraction < 1.0):
        raise ValueError("--validation-fraction must lie in (0,1).")
    if args.tau_power <= 0 or args.tau_fc <= 0:
        raise ValueError("Softmax temperatures must be positive.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.deterministic and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    device = resolve_device(args.device)
    print(f"Device: {device}")
    subjects = scan_dataset(args)
    print(f"Included subjects: {len(subjects)}")

    rows: List[Dict[str, float]] = []
    if args.protocol in ("both", "kfold"):
        for fold, (train_idx, test_idx) in enumerate(
            kfold_splits(subjects, args.n_splits, args.seed), start=1
        ):
            if args.cv_fold is not None and fold != args.cv_fold:
                continue
            rows.append(
                run_split(
                    subjects,
                    train_idx,
                    test_idx,
                    "kfold",
                    f"fold_{fold:02d}",
                    args,
                    device,
                )
            )

    if args.protocol in ("both", "loso"):
        for site, train_idx, test_idx in loso_splits(subjects, args.sites):
            if args.loso_site is not None and site != args.loso_site:
                continue
            rows.append(
                run_split(
                    subjects,
                    train_idx,
                    test_idx,
                    "loso",
                    f"site_{site}",
                    args,
                    device,
                )
            )

    if not rows:
        raise RuntimeError("No evaluation splits were selected.")
    summarize(rows, args.outdir)
    print(f"\nFinished. Results: {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
