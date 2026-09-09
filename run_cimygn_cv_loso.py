#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Leakage-safe subject-level K-fold CV and LOSO evaluation for CiMyGn."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from cimygn_data import (
    CiMyGnDataset,
    Subject,
    fit_full_pca_rotation,
    scan_dataset,
    site_class_strata,
    split_fit_validation,
    standardized_arrays,
)
from cimygn_engine import (
    METRIC_NAMES,
    make_loader,
    metrics_from_predictions,
    predict_loader,
    resolve_device,
    stratified_bootstrap_intervals,
    train_select_restart,
)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            jsonable(payload),
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )


def stable_hash(payload: Any) -> str:
    text = json.dumps(
        jsonable(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def implementation_fingerprint() -> Dict[str, str]:
    source_directory = Path(__file__).resolve().parent
    names = (
        "cimygn_model.py",
        "cimygn_data.py",
        "cimygn_engine.py",
        "run_cimygn_cv_loso.py",
    )
    return {
        name: hashlib.sha256((source_directory / name).read_bytes()).hexdigest()
        for name in names
    }


def runtime_fingerprint() -> Dict[str, str]:
    versions = {}
    for distribution in ("torch", "numpy", "pandas", "scipy", "scikit-learn", "h5py"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    cuda_available = torch.cuda.is_available()
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": str(cuda_available),
        "torch_cuda": str(torch.version.cuda),
        **versions,
    }
    if cuda_available:
        result["cuda_device_0"] = torch.cuda.get_device_name(0)
    return result


def model_config(args) -> Dict[str, Any]:
    return {
        "data_dim": args.n_rois,
        "n_power_modes": args.k_power_modes,
        "n_fc_modes": args.q_fc_modes,
        "gin_hidden": args.gin_hidden,
        "roi_direct_hidden": args.roi_direct_hidden,
        "rnn_hidden": args.rnn_hidden,
        "rnn_layers": args.rnn_layers,
        "static_fc_hidden": args.static_fc_hidden,
        "classifier_hidden": args.classifier_hidden,
        "classifier_input": args.classifier_input,
        "dropout": args.dropout,
        "num_classes": 2,
        "min_scale": args.min_scale,
        "min_cholesky": args.min_cholesky,
        "covariance_jitter": args.covariance_jitter,
        "free_bits": args.free_bits,
        "prior_separation_margin": args.prior_separation_margin,
        "coefficient_temperature": args.coefficient_temperature,
    }


def analysis_signature_payload(subjects: Sequence[Subject], args) -> Dict[str, Any]:
    fields = (
        "sites",
        "n_splits",
        "n_rois",
        "timepoints",
        "standardization_eps",
        "max_constant_rois",
        "strict_data",
        "pca",
        "pca_rank_rtol",
        "graph_top_k",
        "validation_fraction",
        "k_power_modes",
        "q_fc_modes",
        "gin_hidden",
        "roi_direct_hidden",
        "rnn_hidden",
        "rnn_layers",
        "static_fc_hidden",
        "classifier_hidden",
        "classifier_input",
        "dropout",
        "batch_size",
        "sampling_strategy",
        "epochs",
        "min_epochs",
        "patience",
        "min_delta",
        "n_restarts",
        "learning_rate",
        "weight_decay",
        "max_grad_norm",
        "nll_time_samples",
        "nll_weight",
        "beta_kl",
        "lambda_classification",
        "lambda_mode_classification",
        "classification_warmup_epochs",
        "generative_ramp_epochs",
        "free_bits",
        "prior_separation_margin",
        "prior_separation_weight",
        "mode_diversity_weight",
        "occupancy_entropy_weight",
        "occupancy_balance_weight",
        "coefficient_temperature",
        "collapse_probability_sd",
        "threshold_strategy",
        "min_scale",
        "min_cholesky",
        "covariance_jitter",
        "device",
        "torch_threads",
        "deterministic",
        "seed",
    )
    return {
        "analysis": {field: getattr(args, field) for field in fields},
        "implementation_sha256": implementation_fingerprint(),
        "runtime": runtime_fingerprint(),
        "subjects": [
            {
                "subject_id": subject.subject_id,
                "site": subject.site,
                "label": subject.label,
                "content_sha256": subject.content_sha256,
            }
            for subject in subjects
        ],
    }


def split_signature_payload(
    fit_subjects: Sequence[Subject],
    validation_subjects: Sequence[Subject],
    test_subjects: Sequence[Subject],
    protocol: str,
    split_name: str,
    args,
) -> Dict[str, Any]:
    return {
        "experiment_signature": args.experiment_signature,
        "protocol": protocol,
        "split": split_name,
        "fit": [(subject.subject_id, subject.content_sha256) for subject in fit_subjects],
        "validation": [
            (subject.subject_id, subject.content_sha256)
            for subject in validation_subjects
        ],
        "test": [(subject.subject_id, subject.content_sha256) for subject in test_subjects],
    }


def make_split_manifest(
    fit_subjects: Sequence[Subject],
    validation_subjects: Sequence[Subject],
    test_subjects: Sequence[Subject],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "role": role,
                "subject_id": subject.subject_id,
                "path": str(subject.path),
                "site": subject.site,
                "group": subject.group,
                "label": subject.label,
                "content_sha256": subject.content_sha256,
            }
            for role, collection in (
                ("fit", fit_subjects),
                ("validation", validation_subjects),
                ("test", test_subjects),
            )
            for subject in collection
        ]
    )


def outer_kfold_splits(
    subjects: Sequence[Subject], args
) -> List[Tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(len(subjects))
    strata = site_class_strata(subjects)
    counts = pd.Series(strata).value_counts()
    if int(counts.min()) < args.n_splits:
        raise RuntimeError(
            f"Site-by-class stratified {args.n_splits}-fold CV is infeasible: "
            f"the smallest cell has {int(counts.min())} subjects."
        )
    splitter = StratifiedKFold(
        n_splits=args.n_splits, shuffle=True, random_state=args.seed
    )
    return list(splitter.split(indices, strata))


def prepare_pca(
    fit_subjects: Sequence[Subject], args
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if args.pca:
        return fit_full_pca_rotation(
            standardized_arrays(fit_subjects, args.standardization_eps),
            args.pca_rank_rtol,
        )
    rotation = np.eye(args.n_rois, dtype=np.float32)
    eigenvalues = np.ones(args.n_rois, dtype=np.float64)
    diagnostics = {
        "pca_enabled": False,
        "pca_components": args.n_rois,
        "pca_effective_rank": args.n_rois,
        "pca_orthogonality_max_error": 0.0,
    }
    return rotation, eigenvalues, diagnostics


def attach_subject_metadata(
    table: pd.DataFrame, subjects: Sequence[Subject]
) -> pd.DataFrame:
    metadata = pd.DataFrame(
        [
            {
                "subject_id": subject.subject_id,
                "path": str(subject.path),
                "group": subject.group,
            }
            for subject in subjects
        ]
    )
    return table.merge(metadata, on="subject_id", how="left", validate="one_to_one")


def save_mode_parameters(model, pca_rotation: np.ndarray, split_dir: Path) -> None:
    model.eval()
    with torch.no_grad():
        power_means_pca = model.power_means.detach().cpu().numpy()
        power_means_roi = power_means_pca @ pca_rotation.T
        power_scales_pca = model.positive_power_scales().detach().cpu().numpy()
        fc_correlation_pca = model.fc_correlation_modes().detach().cpu().numpy()
        fc_structure_roi = np.einsum(
            "di,qij,ej->qde",
            pca_rotation,
            fc_correlation_pca,
            pca_rotation,
            optimize=True,
        )
    np.savez_compressed(
        split_dir / "mode_parameters.npz",
        power_means_pca=power_means_pca,
        power_means_roi=power_means_roi,
        power_scales_pca=power_scales_pca,
        fc_correlation_pca=fc_correlation_pca,
        fc_structure_roi=fc_structure_roi,
        pca_rotation=pca_rotation,
    )


def run_split(
    subjects: Sequence[Subject],
    outer_train_index: np.ndarray,
    test_index: np.ndarray,
    protocol: str,
    split_name: str,
    args,
    device: torch.device,
) -> Dict[str, Any]:
    split_dir = args.outdir / protocol / split_name
    outer_train_subjects = [subjects[int(index)] for index in outer_train_index]
    test_subjects = [subjects[int(index)] for index in test_index]
    split_seed = args.seed + int(
        hashlib.sha256(f"{protocol}/{split_name}".encode("utf-8")).hexdigest()[:8], 16
    ) % 1_000_000
    fit_relative, validation_relative, validation_scheme = split_fit_validation(
        outer_train_subjects, args.validation_fraction, split_seed
    )
    fit_subjects = [outer_train_subjects[int(index)] for index in fit_relative]
    validation_subjects = [
        outer_train_subjects[int(index)] for index in validation_relative
    ]
    signature_payload = split_signature_payload(
        fit_subjects,
        validation_subjects,
        test_subjects,
        protocol,
        split_name,
        args,
    )
    split_signature = stable_hash(signature_payload)
    metadata_path = split_dir / "metadata.json"

    if split_dir.exists():
        existing = None
        if metadata_path.is_file():
            try:
                existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                existing = None
        if existing and existing.get("status") == "complete":
            if existing.get("split_signature") == split_signature and args.resume:
                print(f"[RESUME] Reusing completed {protocol}/{split_name}")
                return {
                    "Protocol": protocol,
                    "Split": split_name,
                    **existing["metrics"],
                }
            if not args.overwrite_split:
                raise RuntimeError(
                    f"Completed output already exists at {split_dir}. Use a new --outdir "
                    "or explicitly add --overwrite-split."
                )
        if args.overwrite_split or args.resume:
            print(f"[RERUN] Removing incomplete/selected output: {split_dir}")
            shutil.rmtree(split_dir)
        else:
            raise RuntimeError(
                f"Output exists at {split_dir}; use --resume or --overwrite-split."
            )
    split_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n[{protocol.upper()} | {split_name}] fit={len(fit_subjects)}, "
        f"validation={len(validation_subjects)}, test={len(test_subjects)}"
    )
    make_split_manifest(
        fit_subjects, validation_subjects, test_subjects
    ).to_csv(split_dir / "split_manifest.csv", index=False)
    write_json(split_dir / "split_signature_payload.json", signature_payload)

    try:
        pca_rotation, eigenvalues, pca_diagnostics = prepare_pca(
            fit_subjects, args
        )
        pca_diagnostics["pca_enabled"] = bool(args.pca)
        np.savez_compressed(
            split_dir / "training_only_pca.npz",
            rotation=pca_rotation,
            eigenvalues=eigenvalues,
        )
        write_json(split_dir / "preprocessing_diagnostics.json", pca_diagnostics)

        fit_dataset = CiMyGnDataset(
            fit_subjects, args.standardization_eps, args.graph_top_k
        )
        validation_dataset = CiMyGnDataset(
            validation_subjects, args.standardization_eps, args.graph_top_k
        )
        test_dataset = CiMyGnDataset(
            test_subjects, args.standardization_eps, args.graph_top_k
        )

        model, selected, restart_summary, restart_results = train_select_restart(
            model_config(args),
            pca_rotation,
            fit_dataset,
            validation_dataset,
            fit_subjects,
            args,
            device,
            split_seed,
        )
        restart_summary.to_csv(split_dir / "restart_selection.csv", index=False)
        for restart, result in enumerate(restart_results, start=1):
            pd.DataFrame(result["history"]).to_csv(
                split_dir / f"restart_{restart:02d}_history.csv", index=False
            )
        selected_validation_predictions = attach_subject_metadata(
            selected["validation_predictions"], validation_subjects
        )
        selected_validation_predictions.to_csv(
            split_dir / "validation_predictions.csv", index=False
        )
        selected_validation_features = attach_subject_metadata(
            selected["validation_features"], validation_subjects
        )
        selected_validation_features.to_csv(
            split_dir / "validation_dynamic_features.csv", index=False
        )

        test_loader = make_loader(
            test_dataset,
            args.batch_size,
            False,
            args.num_workers,
            split_seed,
            device,
        )
        predictions, dynamic_features = predict_loader(model, test_loader, device)
        threshold = float(selected["threshold"])
        mode_threshold = float(selected["mode_threshold"])
        predictions["threshold"] = threshold
        predictions["y_pred"] = (predictions["prob_MDD"] >= threshold).astype(int)
        predictions["threshold_modes_only"] = mode_threshold
        predictions["y_pred_modes_only"] = (
            predictions["prob_MDD_modes_only"] >= mode_threshold
        ).astype(int)
        predictions.insert(0, "Split", split_name)
        predictions.insert(0, "Protocol", protocol)
        predictions = attach_subject_metadata(predictions, test_subjects)
        predictions.to_csv(split_dir / "predictions.csv", index=False)
        dynamic_features = attach_subject_metadata(dynamic_features, test_subjects)
        dynamic_features.to_csv(
            split_dir / "test_dynamic_features.csv", index=False
        )

        metrics = metrics_from_predictions(
            predictions["y_true"].to_numpy(dtype=int),
            predictions["prob_MDD"].to_numpy(dtype=float),
            predictions["y_pred"].to_numpy(dtype=int),
        )
        mode_metrics = metrics_from_predictions(
            predictions["y_true"].to_numpy(dtype=int),
            predictions["prob_MDD_modes_only"].to_numpy(dtype=float),
            predictions["y_pred_modes_only"].to_numpy(dtype=int),
        )
        save_mode_parameters(model, pca_rotation, split_dir)
        if args.save_models:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config(args),
                    "pca_rotation": pca_rotation,
                    "threshold": threshold,
                    "mode_threshold": mode_threshold,
                    "selected_restart_seed": selected["seed"],
                    "selected_best_epoch": selected["best_epoch"],
                },
                split_dir / "cimygn_model.pt",
            )

        metadata = {
            "status": "complete",
            "experiment_signature": args.experiment_signature,
            "split_signature": split_signature,
            "protocol": protocol,
            "split": split_name,
            "split_seed": split_seed,
            "validation_scheme": validation_scheme,
            "n_fit": len(fit_subjects),
            "n_validation": len(validation_subjects),
            "n_test": len(test_subjects),
            "selected_restart_seed": selected["seed"],
            "selected_best_epoch": selected["best_epoch"],
            "selected_threshold": threshold,
            "selected_mode_threshold": mode_threshold,
            "validation_metrics": selected["validation_metrics"],
            "validation_mode_metrics": selected["validation_mode_metrics"],
            "pca_diagnostics": pca_diagnostics,
            "test_prediction_path_uses_labels": False,
            "metrics": metrics,
            "modes_only_metrics": mode_metrics,
        }
        write_json(metadata_path, metadata)
        print(
            f"[RESULT] {protocol}/{split_name}: "
            f"BA={metrics['Balanced-Accuracy']:.4f}, "
            f"AUC={metrics['ROC-AUC']:.4f}, PR-AUC={metrics['PR-AUC']:.4f}; "
            f"modes-only AUC={mode_metrics['ROC-AUC']:.4f}"
        )
        del model, fit_dataset, validation_dataset, test_dataset
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return {"Protocol": protocol, "Split": split_name, **metrics}
    except Exception:
        write_json(
            split_dir / "failure.json",
            {
                "status": "failed",
                "experiment_signature": args.experiment_signature,
                "split_signature": split_signature,
                "traceback": traceback.format_exc(),
            },
        )
        raise


def completed_artifacts(args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    prediction_tables = []
    for protocol in ("kfold", "loso"):
        protocol_directory = args.outdir / protocol
        if not protocol_directory.is_dir():
            continue
        for metadata_path in sorted(protocol_directory.glob("*/metadata.json")):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                continue
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
            prediction_path = metadata_path.parent / "predictions.csv"
            if prediction_path.is_file():
                prediction_tables.append(pd.read_csv(prediction_path))
    metrics = pd.DataFrame(metric_rows)
    predictions = (
        pd.concat(prediction_tables, ignore_index=True)
        if prediction_tables
        else pd.DataFrame()
    )
    return metrics, predictions


def summarize_completed(args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metrics, predictions = completed_artifacts(args)
    if metrics.empty:
        return pd.DataFrame(), pd.DataFrame()
    metrics = metrics.sort_values(["Protocol", "Split"]).reset_index(drop=True)
    metrics.to_csv(args.outdir / "cimygn_per_split_metrics.csv", index=False)
    macro_rows = []
    pooled_rows = []
    modes_only_pooled_rows = []
    for protocol in ("kfold", "loso"):
        protocol_metrics = metrics[metrics["Protocol"] == protocol]
        protocol_predictions = predictions[predictions["Protocol"] == protocol]
        if protocol_metrics.empty:
            continue
        macro = {"Protocol": protocol, "n_splits": len(protocol_metrics)}
        for metric in METRIC_NAMES:
            macro[f"{metric}_mean"] = float(protocol_metrics[metric].mean())
            macro[f"{metric}_sd"] = (
                float(protocol_metrics[metric].std(ddof=1))
                if len(protocol_metrics) > 1
                else 0.0
            )
        macro_rows.append(macro)
        if protocol_predictions.empty:
            continue
        if protocol_predictions["subject_id"].duplicated().any():
            raise RuntimeError(f"Duplicate pooled predictions detected for {protocol}.")
        pooled_metrics = metrics_from_predictions(
            protocol_predictions["y_true"].to_numpy(dtype=int),
            protocol_predictions["prob_MDD"].to_numpy(dtype=float),
            protocol_predictions["y_pred"].to_numpy(dtype=int),
        )
        intervals = stratified_bootstrap_intervals(
            protocol_predictions,
            args.bootstrap_iterations,
            args.seed + (17 if protocol == "kfold" else 29),
        )
        pooled_rows.append(
            {
                "Protocol": protocol,
                "n_completed_splits": len(protocol_metrics),
                "n_pooled_subjects": len(protocol_predictions),
                **pooled_metrics,
                **intervals,
            }
        )
        if {"prob_MDD_modes_only", "y_pred_modes_only"}.issubset(
            protocol_predictions.columns
        ):
            modes_only_metrics = metrics_from_predictions(
                protocol_predictions["y_true"].to_numpy(dtype=int),
                protocol_predictions["prob_MDD_modes_only"].to_numpy(dtype=float),
                protocol_predictions["y_pred_modes_only"].to_numpy(dtype=int),
            )
            modes_only_pooled_rows.append(
                {
                    "Protocol": protocol,
                    "n_completed_splits": len(protocol_metrics),
                    "n_pooled_subjects": len(protocol_predictions),
                    **modes_only_metrics,
                }
            )
    macro_table = pd.DataFrame(macro_rows)
    pooled_table = pd.DataFrame(pooled_rows)
    macro_table.to_csv(args.outdir / "cimygn_summary_macro.csv", index=False)
    pooled_table.to_csv(args.outdir / "cimygn_summary_pooled.csv", index=False)
    pd.DataFrame(modes_only_pooled_rows).to_csv(
        args.outdir / "cimygn_modes_only_summary_pooled.csv", index=False
    )
    predictions.to_csv(args.outdir / "cimygn_predictions_pooled.csv", index=False)
    print("\n================ CiMyGn SUMMARY ================")
    for _, row in pooled_table.iterrows():
        print(
            f"{row['Protocol']}: pooled n={int(row['n_pooled_subjects'])}, "
            f"BA={row['Balanced-Accuracy']:.4f}, AUC={row['ROC-AUC']:.4f}, "
            f"PR-AUC={row['PR-AUC']:.4f} "
            f"(prevalence={row['MDD-Prevalence']:.4f}); "
            f"splits={int(row['n_completed_splits'])}"
        )
    return macro_table, pooled_table


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="CiMyGn subject-level K-fold CV and LOSO evaluation.",
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
    parser.add_argument(
        "--outdir", type=Path, default=Path("results_cimygn_v2_roi_fused")
    )
    parser.add_argument("--seed", type=int, default=42)

    data = parser.add_argument_group("data and preprocessing")
    data.add_argument("--n-rois", type=int, default=90)
    data.add_argument("--timepoints", type=int, default=200)
    data.add_argument("--standardization-eps", type=float, default=1e-8)
    data.add_argument(
        "--max-constant-rois",
        type=int,
        default=20,
        help=(
            "QC ceiling. The known dataset subject with 15 constant ROIs is retained; "
            "set 0 for the strictest analysis."
        ),
    )
    data.add_argument("--pca", action=argparse.BooleanOptionalAction, default=True)
    data.add_argument("--pca-rank-rtol", type=float, default=1e-10)
    data.add_argument("--graph-top-k", type=int, default=10)
    data.add_argument(
        "--strict-data", action=argparse.BooleanOptionalAction, default=True
    )

    model = parser.add_argument_group("CiMyGn model")
    model.add_argument("--k-power-modes", type=int, default=3)
    model.add_argument("--q-fc-modes", type=int, default=6)
    model.add_argument("--gin-hidden", type=int, default=32)
    model.add_argument("--roi-direct-hidden", type=int, default=64)
    model.add_argument("--rnn-hidden", type=int, default=64)
    model.add_argument("--rnn-layers", type=int, default=1)
    model.add_argument("--static-fc-hidden", type=int, default=64)
    model.add_argument("--classifier-hidden", type=int, default=64)
    model.add_argument(
        "--classifier-input",
        choices=["fused", "modes-only", "encoder-only", "static-only"],
        default="fused",
        help="Classifier ablation. fused is the recommended ROI-aware V2 model.",
    )
    model.add_argument("--dropout", type=float, default=0.2)
    model.add_argument("--min-scale", type=float, default=1e-3)
    model.add_argument("--min-cholesky", type=float, default=1e-3)
    model.add_argument("--covariance-jitter", type=float, default=1e-4)
    model.add_argument("--free-bits", type=float, default=0.02)
    model.add_argument("--prior-separation-margin", type=float, default=0.5)
    model.add_argument("--coefficient-temperature", type=float, default=0.75)

    training = parser.add_argument_group("training")
    training.add_argument("--validation-fraction", type=float, default=0.2)
    training.add_argument("--batch-size", type=int, default=8)
    training.add_argument(
        "--sampling-strategy",
        choices=["site-class", "class", "uniform"],
        default="site-class",
        help=(
            "Training-only sampling. site-class prevents the largest site from "
            "dominating all three loss terms."
        ),
    )
    training.add_argument("--epochs", type=int, default=60)
    training.add_argument("--min-epochs", type=int, default=20)
    training.add_argument("--patience", type=int, default=10)
    training.add_argument("--n-restarts", type=int, default=2)
    training.add_argument("--learning-rate", type=float, default=3e-4)
    training.add_argument("--weight-decay", type=float, default=1e-4)
    training.add_argument("--max-grad-norm", type=float, default=1.0)
    training.add_argument(
        "--nll-time-samples",
        type=int,
        default=32,
        help="Uniform time points used for each training NLL; 0 uses all time points.",
    )
    training.add_argument("--nll-weight", type=float, default=0.25)
    training.add_argument("--beta-kl", type=float, default=0.02)
    training.add_argument("--lambda-classification", type=float, default=1.0)
    training.add_argument("--lambda-mode-classification", type=float, default=0.25)
    training.add_argument("--classification-warmup-epochs", type=int, default=5)
    training.add_argument("--generative-ramp-epochs", type=int, default=10)
    training.add_argument("--prior-separation-weight", type=float, default=0.01)
    training.add_argument("--mode-diversity-weight", type=float, default=0.01)
    training.add_argument("--occupancy-entropy-weight", type=float, default=0.002)
    training.add_argument("--occupancy-balance-weight", type=float, default=0.01)
    training.add_argument(
        "--collapse-probability-sd",
        type=float,
        default=1e-3,
        help="Print a warning when validation probability SD falls below this value.",
    )
    training.add_argument("--min-delta", type=float, default=1e-4)
    training.add_argument(
        "--threshold-strategy",
        choices=["validation-balanced", "fixed-0.5"],
        default="validation-balanced",
    )

    execution = parser.add_argument_group("execution")
    execution.add_argument("--device", default="auto")
    execution.add_argument("--num-workers", type=int, default=0)
    execution.add_argument("--torch-threads", type=int, default=0)
    execution.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    execution.add_argument("--bootstrap-iterations", type=int, default=2000)
    execution.add_argument("--cv-fold", type=int, default=None)
    execution.add_argument("--loso-site", type=int, default=None)
    execution.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    execution.add_argument("--overwrite-split", action="store_true")
    execution.add_argument("--save-models", action="store_true")
    execution.add_argument("--check-only", action="store_true")
    return parser.parse_args(argv)


def validate_args(args):
    args.data_root = args.data_root.expanduser().resolve()
    args.outdir = args.outdir.expanduser().resolve()
    args.sites = list(dict.fromkeys(args.sites))
    if len(args.sites) < 2:
        raise ValueError("At least two sites are required.")
    if args.n_splits < 2:
        raise ValueError("--n-splits must be at least 2.")
    if args.cv_fold is not None and not 1 <= args.cv_fold <= args.n_splits:
        raise ValueError("--cv-fold must be between 1 and --n-splits.")
    if args.loso_site is not None and args.loso_site not in args.sites:
        raise ValueError("--loso-site must also be listed in --sites.")
    if args.n_rois < 2 or args.timepoints < 2:
        raise ValueError("--n-rois and --timepoints must be at least 2.")
    if args.standardization_eps <= 0 or args.pca_rank_rtol <= 0:
        raise ValueError("Standardization/PCA tolerances must be positive.")
    if not 0 <= args.max_constant_rois <= args.n_rois:
        raise ValueError("--max-constant-rois must be between 0 and --n-rois.")
    if not 0 <= args.graph_top_k < args.n_rois:
        raise ValueError("--graph-top-k must be in [0,n_rois).")
    if args.k_power_modes < 2 or args.q_fc_modes < 2:
        raise ValueError("K and Q must both be at least 2.")
    if (
        args.rnn_layers < 1
        or args.gin_hidden < 1
        or args.roi_direct_hidden < 1
        or args.rnn_hidden < 1
        or args.static_fc_hidden < 1
        or args.classifier_hidden < 1
    ):
        raise ValueError("Hidden dimensions/layers must be positive.")
    if not 0 <= args.dropout < 1:
        raise ValueError("--dropout must be in [0,1).")
    if not 0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be in (0,0.5).")
    if args.batch_size < 1 or args.epochs < 1 or args.n_restarts < 1:
        raise ValueError("Batch size, epochs, and restarts must be positive.")
    if not 1 <= args.min_epochs <= args.epochs:
        raise ValueError("--min-epochs must be between 1 and --epochs.")
    if args.patience < 1 or args.learning_rate <= 0:
        raise ValueError("Patience and learning rate must be positive.")
    if (
        args.min_delta < 0
        or args.weight_decay < 0
        or args.classification_warmup_epochs < 0
        or args.generative_ramp_epochs < 0
    ):
        raise ValueError("Delta/decay/warmup parameters cannot be negative.")
    if args.nll_time_samples < 0 or args.nll_time_samples > args.timepoints:
        raise ValueError("--nll-time-samples must be in [0,timepoints].")
    loss_weights = (
        args.nll_weight,
        args.beta_kl,
        args.lambda_classification,
        args.lambda_mode_classification,
        args.prior_separation_weight,
        args.mode_diversity_weight,
        args.occupancy_entropy_weight,
        args.occupancy_balance_weight,
    )
    if min(loss_weights) < 0:
        raise ValueError("Loss weights cannot be negative.")
    if sum(loss_weights) == 0:
        raise ValueError("At least one loss weight must be positive.")
    if args.max_grad_norm <= 0 or args.covariance_jitter <= 0:
        raise ValueError("Gradient/covariance stability parameters must be positive.")
    if args.min_scale <= 0 or args.min_cholesky <= 0:
        raise ValueError("Scale and Cholesky floors must be positive.")
    if args.free_bits < 0 or args.prior_separation_margin < 0:
        raise ValueError("Free bits and prior margin cannot be negative.")
    if args.coefficient_temperature <= 0:
        raise ValueError("--coefficient-temperature must be positive.")
    if args.collapse_probability_sd < 0:
        raise ValueError("--collapse-probability-sd cannot be negative.")
    if args.bootstrap_iterations < 0 or args.num_workers < 0:
        raise ValueError("Bootstrap iterations/workers cannot be negative.")
    args.outdir.mkdir(parents=True, exist_ok=True)
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = validate_args(parse_args(argv))
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    torch.set_float32_matmul_precision("highest")
    print("Python:", sys.version.replace("\n", " "))
    print("PyTorch:", torch.__version__)
    print("Data root:", args.data_root)
    print("Output:", args.outdir)
    print("Sites:", args.sites)
    print(
        "CiMyGn V2:",
        f"classifier={args.classifier_input}; warmup={args.classification_warmup_epochs}; "
        f"ramp={args.generative_ramp_epochs}; max NLL/KL weights="
        f"{args.nll_weight:g}/{args.beta_kl:g}",
    )

    subjects = scan_dataset(args)
    signature_payload = analysis_signature_payload(subjects, args)
    args.experiment_signature = stable_hash(signature_payload)
    run_config_path = args.outdir / "run_config.json"
    if run_config_path.is_file():
        existing = json.loads(run_config_path.read_text(encoding="utf-8"))
        has_splits = any((args.outdir / name).is_dir() for name in ("kfold", "loso"))
        if (
            existing.get("experiment_signature") not in (None, args.experiment_signature)
            and has_splits
        ):
            raise RuntimeError(
                "The output directory contains a different experiment. Use a new --outdir."
            )
    write_json(
        run_config_path,
        {
            "experiment_signature": args.experiment_signature,
            "command": sys.argv,
            "arguments": vars(args),
            "signature_payload": signature_payload,
        },
    )
    if args.check_only:
        print("\nDataset preflight completed successfully; no model was trained.")
        return 0

    device = resolve_device(args.device)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device))
    labels = np.asarray([subject.label for subject in subjects], dtype=int)
    sites = np.asarray([subject.site for subject in subjects], dtype=int)
    indices = np.arange(len(subjects))
    executed = 0

    if args.protocol in ("both", "kfold"):
        for fold, (train_index, test_index) in enumerate(
            outer_kfold_splits(subjects, args), start=1
        ):
            if args.cv_fold is not None and fold != args.cv_fold:
                continue
            run_split(
                subjects,
                train_index,
                test_index,
                "kfold",
                f"fold_{fold:02d}",
                args,
                device,
            )
            executed += 1
            summarize_completed(args)

    if args.protocol in ("both", "loso"):
        for held_site in args.sites:
            if args.loso_site is not None and held_site != args.loso_site:
                continue
            test_index = indices[sites == held_site]
            train_index = indices[sites != held_site]
            if len(np.unique(labels[test_index])) != 2:
                raise RuntimeError(f"Held-out site {held_site} does not contain both classes.")
            run_split(
                subjects,
                train_index,
                test_index,
                "loso",
                f"site_{held_site}",
                args,
                device,
            )
            executed += 1
            summarize_completed(args)

    if executed == 0:
        raise RuntimeError("No split was selected.")
    summarize_completed(args)
    print("\nCompleted. Results:", args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
