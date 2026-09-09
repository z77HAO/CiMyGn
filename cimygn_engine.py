#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Training, deterministic inference, threshold selection, and metrics."""

from __future__ import annotations

import copy
import gc
import math
import random
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from torch.utils.data import DataLoader, WeightedRandomSampler

from cimygn_data import CiMyGnDataset, Subject, collate_subjects
from cimygn_model import CiMyGn


METRIC_NAMES = (
    "Accuracy",
    "Balanced-Accuracy",
    "Sensitivity",
    "Specificity",
    "F1-score",
    "ROC-AUC",
    "PR-AUC",
    "MCC",
)


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return device


def make_loader(
    dataset: CiMyGnDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    device: torch.device,
    sample_weights: Optional[Sequence[float]] = None,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = None
    if sample_weights is not None:
        if len(sample_weights) != len(dataset):
            raise ValueError("Sampling weights do not match the dataset length.")
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_subjects,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def subject_sampling_weights(
    subjects: Sequence[Subject], strategy: str
) -> Optional[np.ndarray]:
    if strategy == "uniform":
        return None
    if strategy == "class":
        keys = [str(subject.label) for subject in subjects]
    elif strategy == "site-class":
        keys = [f"S{subject.site}_Y{subject.label}" for subject in subjects]
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    counts = Counter(keys)
    weights = np.asarray([1.0 / counts[key] for key in keys], dtype=np.float64)
    return weights / weights.mean()


def class_weight_tensor(labels: Sequence[int], device: torch.device) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=2).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"Both classes are required for training; counts={counts.tolist()}")
    weights = len(labels) / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def safe_auc(y_true: np.ndarray, probability: np.ndarray) -> float:
    if len(np.unique(y_true)) != 2:
        return float("nan")
    return float(roc_auc_score(y_true, probability))


def metrics_from_predictions(
    y_true: np.ndarray,
    probability: np.ndarray,
    prediction: np.ndarray,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=int)
    if not (len(y_true) == len(probability) == len(prediction)):
        raise ValueError("Prediction arrays have different lengths.")
    if not np.all(np.isfinite(probability)):
        raise FloatingPointError("Predicted probabilities contain NaN/Inf.")
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    return {
        "Accuracy": float(accuracy_score(y_true, prediction)),
        "Balanced-Accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "F1-score": float(f1_score(y_true, prediction, zero_division=0)),
        "ROC-AUC": safe_auc(y_true, probability),
        "PR-AUC": float(average_precision_score(y_true, probability)),
        "MDD-Prevalence": float(y_true.mean()),
        "MCC": float(matthews_corrcoef(y_true, prediction)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "n_test": int(len(y_true)),
    }


def select_balanced_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    unique = np.unique(np.asarray(probability, dtype=np.float64))
    midpoints = (
        0.5 * (unique[:-1] + unique[1:])
        if len(unique) > 1
        else np.array([], dtype=np.float64)
    )
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], unique, midpoints)))
    scores = np.asarray(
        [
            balanced_accuracy_score(y_true, (probability >= threshold).astype(int))
            for threshold in candidates
        ]
    )
    best = float(np.max(scores))
    tied = candidates[np.isclose(scores, best, rtol=0.0, atol=1e-12)]
    return float(tied[np.argmin(np.abs(tied - 0.5))])


def coefficient_feature_names(n_power_modes: int, n_fc_modes: int) -> List[str]:
    names = []
    for prefix, n_modes in (("power", n_power_modes), ("fc", n_fc_modes)):
        for statistic in ("mean", "sd", "mean_abs_diff"):
            names.extend(
                f"{prefix}_{statistic}_mode_{mode}"
                for mode in range(1, n_modes + 1)
            )
    for prefix, n_modes in (("power", n_power_modes), ("fc", n_fc_modes)):
        names.extend(
            f"{prefix}_transition_{source}_to_{target}"
            for source in range(1, n_modes + 1)
            for target in range(1, n_modes + 1)
        )
    names.extend(["power_mean_entropy", "fc_mean_entropy"])
    return names


def coefficient_features(power: torch.Tensor, fc: torch.Tensor) -> torch.Tensor:
    summaries = []
    transitions = []
    entropies = []
    for coefficients in (power, fc):
        mean = coefficients.mean(dim=1)
        standard_deviation = coefficients.std(dim=1, unbiased=False)
        if coefficients.shape[1] > 1:
            mean_absolute_difference = coefficients.diff(dim=1).abs().mean(dim=1)
            transition = torch.einsum(
                "bti,btj->bij", coefficients[:, :-1], coefficients[:, 1:]
            ) / (coefficients.shape[1] - 1)
        else:
            mean_absolute_difference = torch.zeros_like(mean)
            transition = coefficients.new_zeros(
                (coefficients.shape[0], coefficients.shape[2], coefficients.shape[2])
            )
        entropy = -(
            coefficients.clamp_min(1e-8) * coefficients.clamp_min(1e-8).log()
        ).sum(dim=-1).mean(dim=1, keepdim=True)
        summaries.extend([mean, standard_deviation, mean_absolute_difference])
        transitions.append(transition.flatten(start_dim=1))
        entropies.append(entropy)
    return torch.cat([*summaries, *transitions, *entropies], dim=-1)


@torch.no_grad()
def predict_loader(
    model: CiMyGn,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model.eval()
    prediction_rows = []
    feature_rows = []
    feature_names = coefficient_feature_names(
        model.n_power_modes, model.n_fc_modes
    )
    for x, labels, subject_ids, sites, adjacency in loader:
        x = x.to(device, non_blocking=True)
        adjacency = adjacency.to(device, non_blocking=True)
        outputs = model(
            x,
            adjacency,
            labels=None,
            sample_latent=False,
            compute_nll=False,
        )
        probabilities = torch.softmax(outputs["logits"].float(), dim=-1)[:, 1]
        mode_probabilities = torch.softmax(
            outputs["mode_logits"].float(), dim=-1
        )[:, 1]
        features = coefficient_features(
            outputs["power_coefficients"], outputs["fc_coefficients"]
        )
        for index, subject_id in enumerate(subject_ids):
            prediction_rows.append(
                {
                    "subject_id": subject_id,
                    "site": int(sites[index].item()),
                    "y_true": int(labels[index].item()),
                    "prob_MDD": float(probabilities[index].cpu().item()),
                    "prob_MDD_modes_only": float(
                        mode_probabilities[index].cpu().item()
                    ),
                }
            )
            feature_rows.append(
                {
                    "subject_id": subject_id,
                    "site": int(sites[index].item()),
                    "y_true": int(labels[index].item()),
                    **{
                        name: float(value)
                        for name, value in zip(
                            feature_names, features[index].cpu().tolist()
                        )
                    },
                }
            )
    predictions = pd.DataFrame(prediction_rows)
    dynamic_features = pd.DataFrame(feature_rows)
    if predictions.empty or len(predictions) != len(loader.dataset):
        raise RuntimeError(
            f"Inference produced {len(predictions)} predictions for "
            f"{len(loader.dataset)} subjects."
        )
    return predictions, dynamic_features


def train_one_restart(
    model_config: Dict,
    pca_rotation: np.ndarray,
    fit_loader: DataLoader,
    validation_loader: DataLoader,
    fit_labels: Sequence[int],
    args,
    device: torch.device,
    seed: int,
):
    set_seed(seed, args.deterministic)
    model = CiMyGn(
        model_config,
        torch.from_numpy(np.asarray(pca_rotation, dtype=np.float32)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.learning_rate * 0.05
    )
    class_weights = (
        class_weight_tensor(fit_labels, device)
        if args.sampling_strategy == "uniform"
        else torch.ones(2, dtype=torch.float32, device=device)
    )
    history: List[Dict] = []
    best_state = None
    best_epoch = 0
    best_auc = -math.inf
    best_balanced_accuracy = -math.inf
    epochs_without_improvement = 0
    selection_start_epoch = max(
        1, min(args.classification_warmup_epochs, args.epochs)
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {
            "loss": 0.0,
            "nll": 0.0,
            "kl": 0.0,
            "kl_raw": 0.0,
            "cls": 0.0,
            "mode_cls": 0.0,
            "prior_separation": 0.0,
            "mode_diversity": 0.0,
            "mode_entropy": 0.0,
            "mode_balance": 0.0,
        }
        n_subjects = 0
        current_weights = {
            "generative_fraction": 0.0,
            "nll_weight": 0.0,
            "kl_weight": 0.0,
            "prior_separation_weight": 0.0,
        }
        sampled_probabilities = []
        sampled_labels = []
        compute_nll = bool(
            args.nll_weight > 0.0
            and epoch > args.classification_warmup_epochs
        )
        for x, labels, _, _, adjacency in fit_loader:
            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            adjacency = adjacency.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                x,
                adjacency,
                labels=labels,
                sample_latent=True,
                compute_nll=compute_nll,
                nll_time_samples=args.nll_time_samples,
                class_weights=class_weights,
            )
            loss, current_weights = model.total_loss(
                outputs,
                epoch=epoch,
                nll_weight=args.nll_weight,
                kl_max_weight=args.beta_kl,
                classification_weight=args.lambda_classification,
                mode_classification_weight=args.lambda_mode_classification,
                classification_warmup_epochs=args.classification_warmup_epochs,
                generative_ramp_epochs=args.generative_ramp_epochs,
                prior_separation_weight=args.prior_separation_weight,
                mode_diversity_weight=args.mode_diversity_weight,
                occupancy_entropy_weight=args.occupancy_entropy_weight,
                occupancy_balance_weight=args.occupancy_balance_weight,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss at epoch {epoch}."
                )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(
                    f"Non-finite gradient norm at epoch {epoch}."
                )
            optimizer.step()
            batch_size = len(labels)
            n_subjects += batch_size
            totals["loss"] += float(loss.detach().cpu()) * batch_size
            if outputs["nll"] is not None:
                totals["nll"] += float(outputs["nll"].detach().cpu()) * batch_size
            totals["kl"] += float(outputs["kl"].detach().cpu()) * batch_size
            totals["kl_raw"] += float(outputs["kl_raw"].detach().cpu()) * batch_size
            totals["cls"] += (
                float(outputs["classification_loss"].detach().cpu()) * batch_size
            )
            totals["mode_cls"] += (
                float(outputs["mode_classification_loss"].detach().cpu())
                * batch_size
            )
            for name in (
                "prior_separation",
                "mode_diversity",
                "mode_entropy",
                "mode_balance",
            ):
                totals[name] += float(outputs[name].detach().cpu()) * batch_size
            sampled_probabilities.append(
                torch.softmax(outputs["logits"].detach().float(), dim=-1)[:, 1]
                .cpu()
                .numpy()
            )
            sampled_labels.append(labels.detach().cpu().numpy())
        scheduler.step()

        validation_predictions, validation_features = predict_loader(
            model, validation_loader, device
        )
        y_validation = validation_predictions["y_true"].to_numpy(dtype=int)
        probability = validation_predictions["prob_MDD"].to_numpy(dtype=float)
        validation_prediction = (probability >= 0.5).astype(int)
        validation_auc = safe_auc(y_validation, probability)
        validation_balanced_accuracy = float(
            balanced_accuracy_score(y_validation, validation_prediction)
        )
        validation_threshold = select_balanced_threshold(
            y_validation, probability
        )
        validation_balanced_accuracy_best_threshold = float(
            balanced_accuracy_score(
                y_validation,
                (probability >= validation_threshold).astype(int),
            )
        )
        train_auc = safe_auc(
            np.concatenate(sampled_labels),
            np.concatenate(sampled_probabilities),
        )
        feature_values = validation_features.drop(
            columns=["subject_id", "site", "y_true"]
        ).to_numpy(dtype=float)
        mean_dynamic_feature_sd = float(
            np.std(feature_values, axis=0, ddof=0).mean()
        )
        row = {
            "epoch": epoch,
            "train_total_loss": totals["loss"] / n_subjects,
            "train_nll_per_roi": (
                totals["nll"] / n_subjects if compute_nll else float("nan")
            ),
            "train_kl_per_latent": totals["kl"] / n_subjects,
            "train_kl_raw_per_latent": totals["kl_raw"] / n_subjects,
            "train_classification_loss": totals["cls"] / n_subjects,
            "train_mode_classification_loss": totals["mode_cls"] / n_subjects,
            "train_prior_separation_penalty": totals["prior_separation"] / n_subjects,
            "train_mode_diversity_penalty": totals["mode_diversity"] / n_subjects,
            "train_mode_entropy": totals["mode_entropy"] / n_subjects,
            "train_mode_balance_penalty": totals["mode_balance"] / n_subjects,
            "train_sampled_ROC_AUC": train_auc,
            "generative_fraction": current_weights["generative_fraction"],
            "effective_nll_weight": current_weights["nll_weight"],
            "effective_kl_weight": current_weights["kl_weight"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "validation_ROC_AUC": validation_auc,
            "validation_Balanced_Accuracy_at_0.5": validation_balanced_accuracy,
            "validation_selected_threshold": validation_threshold,
            "validation_Balanced_Accuracy_best_threshold": validation_balanced_accuracy_best_threshold,
            "validation_probability_min": float(probability.min()),
            "validation_probability_max": float(probability.max()),
            "validation_probability_mean": float(probability.mean()),
            "validation_probability_sd": float(probability.std(ddof=0)),
            "validation_positive_fraction_at_0.5": float(
                np.mean(probability >= 0.5)
            ),
            "validation_mean_dynamic_feature_sd": mean_dynamic_feature_sd,
        }
        history.append(row)
        print(
            f"  epoch {epoch:03d}: loss={row['train_total_loss']:.4f}, "
            f"cls={row['train_classification_loss']:.4f}, "
            f"nll={row['train_nll_per_roi']:.4f}, kl_raw={row['train_kl_raw_per_latent']:.4f}, "
            f"train_auc={train_auc:.4f}, val_auc={validation_auc:.4f}, "
            f"val_BA@.5={validation_balanced_accuracy:.4f}, "
            f"p_sd={row['validation_probability_sd']:.4f}, "
            f"gen={row['generative_fraction']:.2f}"
        )
        if (
            epoch >= selection_start_epoch + 2
            and row["validation_probability_sd"] < args.collapse_probability_sd
        ):
            print(
                "  [COLLAPSE WARNING] validation probabilities have near-zero "
                f"dispersion (sd={row['validation_probability_sd']:.6g})."
            )

        eligible = epoch >= selection_start_epoch
        improved = eligible and (
            validation_auc > best_auc + args.min_delta
            or (
                np.isclose(validation_auc, best_auc, atol=args.min_delta, rtol=0.0)
                and validation_balanced_accuracy_best_threshold
                > best_balanced_accuracy + args.min_delta
            )
        )
        if improved:
            best_auc = validation_auc
            best_balanced_accuracy = validation_balanced_accuracy_best_threshold
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        elif eligible:
            epochs_without_improvement += 1
        if epoch >= args.min_epochs and epochs_without_improvement >= args.patience:
            print(f"  early stopping at epoch {epoch}; best epoch={best_epoch}")
            break

    if best_state is None:
        raise RuntimeError("No valid validation checkpoint was selected.")
    model.load_state_dict(best_state)
    validation_predictions, validation_features = predict_loader(
        model, validation_loader, device
    )
    y_validation = validation_predictions["y_true"].to_numpy(dtype=int)
    probability = validation_predictions["prob_MDD"].to_numpy(dtype=float)
    threshold = (
        0.5
        if args.threshold_strategy == "fixed-0.5"
        else select_balanced_threshold(y_validation, probability)
    )
    validation_predictions["threshold"] = threshold
    validation_predictions["y_pred"] = (
        validation_predictions["prob_MDD"] >= threshold
    ).astype(int)
    validation_metrics = metrics_from_predictions(
        y_validation,
        probability,
        validation_predictions["y_pred"].to_numpy(dtype=int),
    )
    mode_probability = validation_predictions["prob_MDD_modes_only"].to_numpy(
        dtype=float
    )
    mode_threshold = (
        0.5
        if args.threshold_strategy == "fixed-0.5"
        else select_balanced_threshold(y_validation, mode_probability)
    )
    validation_predictions["threshold_modes_only"] = mode_threshold
    validation_predictions["y_pred_modes_only"] = (
        mode_probability >= mode_threshold
    ).astype(int)
    validation_mode_metrics = metrics_from_predictions(
        y_validation,
        mode_probability,
        validation_predictions["y_pred_modes_only"].to_numpy(dtype=int),
    )
    result = {
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_metrics": validation_metrics,
        "validation_mode_metrics": validation_mode_metrics,
        "threshold": threshold,
        "mode_threshold": mode_threshold,
        "history": history,
        "state_dict": best_state,
        "validation_predictions": validation_predictions,
        "validation_features": validation_features,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return result


def train_select_restart(
    model_config: Dict,
    pca_rotation: np.ndarray,
    fit_dataset: CiMyGnDataset,
    validation_dataset: CiMyGnDataset,
    fit_subjects: Sequence[Subject],
    args,
    device: torch.device,
    split_seed: int,
):
    fit_labels = [subject.label for subject in fit_subjects]
    fit_sampling_weights = subject_sampling_weights(
        fit_subjects, args.sampling_strategy
    )
    restart_results = []
    summary_rows = []
    for restart in range(1, args.n_restarts + 1):
        seed = split_seed + 1009 * restart
        print(f"[CiMyGn] restart {restart}/{args.n_restarts}; seed={seed}")
        fit_loader = make_loader(
            fit_dataset,
            args.batch_size,
            True,
            args.num_workers,
            seed,
            device,
            fit_sampling_weights,
        )
        validation_loader = make_loader(
            validation_dataset,
            args.batch_size,
            False,
            args.num_workers,
            seed,
            device,
        )
        result = train_one_restart(
            model_config,
            pca_rotation,
            fit_loader,
            validation_loader,
            fit_labels,
            args,
            device,
            seed,
        )
        restart_results.append(result)
        metrics = result["validation_metrics"]
        summary_rows.append(
            {
                "restart": restart,
                "seed": seed,
                "best_epoch": result["best_epoch"],
                "selected_threshold": result["threshold"],
                "selected_mode_threshold": result["mode_threshold"],
                **metrics,
                **{
                    f"modes_only_{name}": value
                    for name, value in result["validation_mode_metrics"].items()
                },
            }
        )

    best_index = max(
        range(len(restart_results)),
        key=lambda index: (
            restart_results[index]["validation_metrics"]["ROC-AUC"],
            restart_results[index]["validation_metrics"]["Balanced-Accuracy"],
            -restart_results[index]["best_epoch"],
        ),
    )
    selected = restart_results[best_index]
    model = CiMyGn(
        model_config,
        torch.from_numpy(np.asarray(pca_rotation, dtype=np.float32)),
    ).to(device)
    model.load_state_dict(selected["state_dict"])
    return model, selected, pd.DataFrame(summary_rows), restart_results


def stratified_bootstrap_intervals(
    predictions: pd.DataFrame,
    iterations: int,
    seed: int,
) -> Dict[str, float]:
    if iterations <= 0:
        return {}
    y = predictions["y_true"].to_numpy(dtype=int)
    probability = predictions["prob_MDD"].to_numpy(dtype=float)
    prediction = predictions["y_pred"].to_numpy(dtype=int)
    sites = predictions["site"].to_numpy(dtype=int)
    strata = np.asarray([f"S{site}_Y{label}" for site, label in zip(sites, y)])
    index_groups = [np.flatnonzero(strata == value) for value in np.unique(strata)]
    rng = np.random.default_rng(seed)
    draws = {metric: [] for metric in METRIC_NAMES}
    for _ in range(iterations):
        sampled = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in index_groups]
        )
        metrics = metrics_from_predictions(
            y[sampled], probability[sampled], prediction[sampled]
        )
        for metric in METRIC_NAMES:
            if np.isfinite(metrics[metric]):
                draws[metric].append(metrics[metric])
    intervals = {}
    for metric, values in draws.items():
        if values:
            intervals[f"{metric}_CI95_low"] = float(np.percentile(values, 2.5))
            intervals[f"{metric}_CI95_high"] = float(np.percentile(values, 97.5))
    return intervals
