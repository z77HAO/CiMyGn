#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Training, inference, and evaluation utilities for manuscript-aligned CiMyGn."""

from __future__ import annotations

import copy
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from cimygn_data import Subject, safe_subject_standardize
from cimygn_model import CiMyGn


METRIC_NAMES = ("Accuracy", "Sensitivity", "Specificity", "F1-score", "ROC-AUC")


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
        raise RuntimeError("CUDA was requested but is not available.")
    return device


class SequenceDataset(Dataset):
    """Participant-level standardized ROI time series; no graph/static-FC branch."""

    def __init__(self, subjects: Sequence[Subject], standardization_eps: float) -> None:
        self.subjects = list(subjects)
        self.arrays = [
            safe_subject_standardize(subject.x, standardization_eps)[0]
            for subject in self.subjects
        ]

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, index: int):
        subject = self.subjects[index]
        return (
            torch.from_numpy(self.arrays[index]),
            torch.tensor(subject.label, dtype=torch.long),
            subject.subject_id,
            torch.tensor(subject.site, dtype=torch.long),
        )


def _collate(batch):
    x, labels, subject_ids, sites = zip(*batch)
    return torch.stack(x), torch.stack(labels), list(subject_ids), torch.stack(sites)


def make_loader(
    dataset: SequenceDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
        collate_fn=_collate,
        generator=generator,
        persistent_workers=int(num_workers) > 0,
    )


def safe_auc(y_true: np.ndarray, probability: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true)) != 2:
        return float("nan")
    return float(roc_auc_score(y_true, probability))


def metrics_from_predictions(
    y_true: np.ndarray,
    probability: np.ndarray,
    prediction: np.ndarray,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = np.asarray(prediction, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    return {
        "Accuracy": float(accuracy_score(y_true, prediction)),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "F1-score": float(f1_score(y_true, prediction, zero_division=0)),
        "ROC-AUC": safe_auc(y_true, probability),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "n_test": int(len(y_true)),
    }


def _majority_filter(labels: np.ndarray) -> np.ndarray:
    """Three-frame majority-vote filter used for ML/MI characterization."""
    labels = np.asarray(labels, dtype=int)
    if len(labels) < 3:
        return labels.copy()
    out = labels.copy()
    for t in range(1, len(labels) - 1):
        window = labels[t - 1 : t + 2]
        values, counts = np.unique(window, return_counts=True)
        winner = values[np.argmax(counts)]
        if counts.max() >= 2:
            out[t] = winner
    return out


def _mode_temporal_statistics(coefficients: np.ndarray) -> Dict[str, float]:
    """FO from continuous coefficients; ML/MI from smoothed argmax states."""
    coefficients = np.asarray(coefficients, dtype=float)
    n_timepoints, n_modes = coefficients.shape
    hard = _majority_filter(np.argmax(coefficients, axis=1))
    result: Dict[str, float] = {}
    for mode in range(n_modes):
        result[f"FO_mode_{mode + 1}"] = float(coefficients[:, mode].mean())

        runs = []
        onsets = []
        t = 0
        while t < n_timepoints:
            if hard[t] != mode:
                t += 1
                continue
            start = t
            onsets.append(start)
            while t < n_timepoints and hard[t] == mode:
                t += 1
            runs.append(t - start)
        result[f"ML_mode_{mode + 1}"] = (
            float(np.mean(runs)) if runs else float("nan")
        )
        result[f"MI_mode_{mode + 1}"] = (
            float(np.mean(np.diff(onsets))) if len(onsets) >= 2 else float("nan")
        )
    return result


@torch.no_grad()
def predict_loader(
    model: CiMyGn,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model.eval()
    prediction_rows: List[Dict] = []
    feature_rows: List[Dict] = []

    for x, labels, subject_ids, sites in loader:
        x = x.to(device, non_blocking=True)
        outputs = model(
            x,
            labels=None,
            sample_latent=False,
            compute_nll=False,
        )
        probabilities_all = torch.softmax(outputs["logits"].float(), dim=-1)
        predictions = torch.argmax(probabilities_all, dim=-1)
        probabilities = probabilities_all[:, 1]
        alpha = outputs["power_coefficients"].cpu().numpy()
        beta = outputs["fc_coefficients"].cpu().numpy()

        for i, subject_id in enumerate(subject_ids):
            prediction_rows.append(
                {
                    "subject_id": subject_id,
                    "site": int(sites[i].item()),
                    "y_true": int(labels[i].item()),
                    "y_pred": int(predictions[i].cpu().item()),
                    "prob_MDD": float(probabilities[i].cpu().item()),
                }
            )
            power_stats = {
                f"power_{key}": value
                for key, value in _mode_temporal_statistics(alpha[i]).items()
            }
            fc_stats = {
                f"fc_{key}": value
                for key, value in _mode_temporal_statistics(beta[i]).items()
            }
            feature_rows.append(
                {
                    "subject_id": subject_id,
                    "site": int(sites[i].item()),
                    "y_true": int(labels[i].item()),
                    **power_stats,
                    **fc_stats,
                }
            )

    predictions_df = pd.DataFrame(prediction_rows)
    features_df = pd.DataFrame(feature_rows)
    if len(predictions_df) != len(loader.dataset):
        raise RuntimeError("Inference did not return one prediction per subject.")
    return predictions_df, features_df


def linear_kl_weight(epoch: int, anneal_epochs: int) -> float:
    """Linear 0->1 KL annealing.

    The manuscript specifies annealing of lambda_KL but does not report the exact
    schedule. This explicit linear schedule is therefore an implementation choice,
    not an additional claim about the manuscript.
    """
    if anneal_epochs <= 0:
        return 1.0
    return float(min(1.0, max(0.0, epoch / float(anneal_epochs))))


def train_model(
    model_config: Dict,
    pca_rotation: np.ndarray,
    fit_dataset: SequenceDataset,
    validation_dataset: SequenceDataset,
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
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    fit_loader = make_loader(
        fit_dataset,
        args.batch_size,
        True,
        args.num_workers,
        seed,
        device,
    )
    validation_loader = make_loader(
        validation_dataset,
        args.batch_size,
        False,
        args.num_workers,
        seed + 1,
        device,
    )

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    best_auc = -math.inf
    epochs_without_improvement = 0
    history: List[Dict] = []

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        lambda_kl = linear_kl_weight(epoch, int(args.kl_anneal_epochs))
        totals = {"loss": 0.0, "nll": 0.0, "kl": 0.0, "cls": 0.0}
        n_subjects = 0

        for x, labels, _, _ in fit_loader:
            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                x,
                labels=labels,
                sample_latent=True,
                compute_nll=True,
                nll_chunk_size=args.nll_chunk_size,
            )
            loss = model.total_loss(
                outputs,
                lambda_kl=lambda_kl,
                gamma_cls=args.gamma_cls,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}.")
            loss.backward()
            optimizer.step()

            batch_size = int(labels.shape[0])
            n_subjects += batch_size
            totals["loss"] += float(loss.detach().cpu()) * batch_size
            totals["nll"] += float(outputs["nll"].detach().cpu()) * batch_size
            totals["kl"] += float(outputs["kl"].detach().cpu()) * batch_size
            totals["cls"] += (
                float(outputs["classification_loss"].detach().cpu()) * batch_size
            )

        val_predictions, _ = predict_loader(model, validation_loader, device)
        val_auc = safe_auc(
            val_predictions["y_true"].to_numpy(dtype=int),
            val_predictions["prob_MDD"].to_numpy(dtype=float),
        )
        row = {
            "epoch": epoch,
            "lambda_kl": lambda_kl,
            "train_loss": totals["loss"] / max(n_subjects, 1),
            "train_nll": totals["nll"] / max(n_subjects, 1),
            "train_kl": totals["kl"] / max(n_subjects, 1),
            "train_cls": totals["cls"] / max(n_subjects, 1),
            "validation_ROC_AUC": val_auc,
        }
        history.append(row)
        print(
            f"  epoch {epoch:03d}: loss={row['train_loss']:.5f}, "
            f"nll={row['train_nll']:.5f}, kl={row['train_kl']:.5f}, "
            f"cls={row['train_cls']:.5f}, lambda_kl={lambda_kl:.3f}, "
            f"val_auc={val_auc:.4f}"
        )

        improved = np.isfinite(val_auc) and val_auc > best_auc + 1e-8
        if improved:
            best_auc = float(val_auc)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= int(args.patience):
            print(f"  early stopping at epoch {epoch}; best epoch={best_epoch}")
            break

    if best_state is None:
        raise RuntimeError("No valid checkpoint was selected from validation ROC-AUC.")
    model.load_state_dict(best_state)

    validation_predictions, validation_features = predict_loader(
        model, validation_loader, device
    )
    validation_metrics = metrics_from_predictions(
        validation_predictions["y_true"].to_numpy(dtype=int),
        validation_predictions["prob_MDD"].to_numpy(dtype=float),
        validation_predictions["y_pred"].to_numpy(dtype=int),
    )
    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_validation_auc": best_auc,
        "history": history,
        "validation_predictions": validation_predictions,
        "validation_features": validation_features,
        "validation_metrics": validation_metrics,
    }
