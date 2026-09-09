#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dataset loading, QC, preprocessing, PCA rotation, and graph construction."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

try:
    import h5py
except Exception:
    h5py = None


SUPPORTED_EXTENSIONS = {".npy", ".npz", ".mat", ".csv", ".txt"}
PREFERRED_MAT_KEYS = (
    "ROISignals",
    "roiSignals",
    "roi_signals",
    "roi_ts",
    "ROITimeSeries",
    "time_series",
    "timeseries",
    "timecourse",
    "X",
    "data",
    "Data",
    "bold",
    "BOLD",
)


@dataclass
class Subject:
    subject_id: str
    path: Path
    site: int
    group: str
    label: int
    x: np.ndarray
    content_sha256: str
    original_shape: Tuple[int, int]
    orientation: str
    n_constant_rois: int


def find_site_directory(root: Path, site: int) -> Path:
    candidates = [
        root / f"rest_meta_fmri_S{site}",
        root / f"REST_meta_fmri_S{site}",
        root / f"rest_meta_fmri_site{site}",
        root / f"Site{site}",
        root / f"site{site}",
        root / f"S{site}",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    target_names = {
        f"rest_meta_fmri_s{site}",
        f"site{site}",
        f"s{site}",
    }
    matches = [
        path
        for path in root.rglob("*")
        if path.is_dir() and path.name.lower() in target_names
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple directories match site {site}:\n"
            + "\n".join(f"  {path}" for path in matches)
        )
    raise FileNotFoundError(f"Cannot find site {site} below {root}")


def find_group_directory(site_directory: Path, group: str) -> Path:
    aliases = {
        "HC": ["HC", "hc", "Control", "Controls", "healthy", "Healthy"],
        "MDD": [
            "MDD",
            "mdd",
            "Mdd",
            "MD",
            "md",
            "Patient",
            "Patients",
            "patient",
            "patients",
        ],
    }[group]
    for alias in aliases:
        candidate = site_directory / alias
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Cannot find {group} directory below {site_directory}")


def load_mat_array(path: Path) -> np.ndarray:
    try:
        mapping = loadmat(path)
        for key in PREFERRED_MAT_KEYS:
            if key in mapping:
                value = np.asarray(mapping[key])
                if value.ndim == 2 and np.issubdtype(value.dtype, np.number):
                    return value
        candidates = []
        for key, value in mapping.items():
            if key.startswith("__"):
                continue
            array = np.asarray(value)
            if (
                array.ndim == 2
                and min(array.shape) > 1
                and np.issubdtype(array.dtype, np.number)
            ):
                candidates.append((key, array))
        if not candidates:
            raise ValueError("No 2-D numeric array found in MAT file.")
        candidates.sort(key=lambda item: item[1].size, reverse=True)
        return candidates[0][1]
    except NotImplementedError:
        if h5py is None:
            raise RuntimeError("MATLAB v7.3 file detected; install h5py.")
        with h5py.File(path, "r") as handle:
            candidates = []

            def visitor(name, obj):
                if (
                    isinstance(obj, h5py.Dataset)
                    and len(obj.shape) == 2
                    and np.issubdtype(obj.dtype, np.number)
                    and not np.issubdtype(obj.dtype, np.complexfloating)
                ):
                    candidates.append((name, obj.shape))

            handle.visititems(visitor)
            if not candidates:
                raise ValueError("No 2-D numeric dataset found in MATLAB v7.3 file.")
            candidates.sort(key=lambda item: int(np.prod(item[1])), reverse=True)
            return np.asarray(handle[candidates[0][0]])


def load_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        value = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            candidates = []
            for key in archive.files:
                array = np.asarray(archive[key])
                if array.ndim == 2 and np.issubdtype(array.dtype, np.number):
                    candidates.append((key, array))
            if not candidates:
                raise ValueError("No 2-D numeric array found in NPZ file.")
            candidates.sort(key=lambda item: item[1].size, reverse=True)
            value = candidates[0][1]
    elif suffix == ".mat":
        value = load_mat_array(path)
    elif suffix == ".csv":
        value = np.loadtxt(path, delimiter=",")
    elif suffix == ".txt":
        try:
            value = np.loadtxt(path)
        except Exception:
            value = np.loadtxt(path, delimiter=",")
    else:
        raise ValueError(f"Unsupported extension: {suffix}")
    value = np.asarray(value)
    if not np.issubdtype(value.dtype, np.number):
        raise ValueError(f"Expected numeric data, found dtype={value.dtype}")
    if np.iscomplexobj(value):
        raise ValueError("Complex-valued time series are not supported.")
    return value


def orient_and_crop(
    value: np.ndarray,
    n_rois: int,
    timepoints: int,
) -> Tuple[np.ndarray, Tuple[int, int], str]:
    array = np.squeeze(value)
    if array.ndim != 2:
        raise ValueError(f"Expected 2-D data, found shape={array.shape}")
    original_shape = (int(array.shape[0]), int(array.shape[1]))
    if array.shape == (n_rois, n_rois):
        raise ValueError(
            f"shape={array.shape} looks like a static FC matrix; ROI time series are required."
        )
    if array.shape[1] == n_rois and array.shape[0] >= timepoints:
        x = array[:timepoints]
        orientation = "time_by_roi"
    elif array.shape[0] == n_rois and array.shape[1] >= timepoints:
        x = array[:, :timepoints].T
        orientation = "roi_by_time_transposed"
    else:
        raise ValueError(
            f"Cannot interpret shape={array.shape}; expected (time,{n_rois}) or "
            f"({n_rois},time), with time >= {timepoints}."
        )
    x = np.asarray(x, dtype=np.float32)
    if not np.all(np.isfinite(x)):
        raise ValueError("NaN/Inf found in time series.")
    return x, original_shape, orientation


def content_hash(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x, dtype=np.float32).view(np.uint8)).hexdigest()


def make_subject_id(root: Path, path: Path, site: int, group: str) -> str:
    try:
        relative = path.relative_to(root).with_suffix("")
    except ValueError:
        relative = path.with_suffix("")
    return f"S{site}_{group}_" + "__".join(relative.parts)


def scan_dataset(args) -> List[Subject]:
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {args.data_root}")
    subjects: List[Subject] = []
    rejected: List[Dict] = []
    constant_records: List[Dict] = []

    for site in args.sites:
        site_directory = find_site_directory(args.data_root, site)
        for group, label in (("HC", 0), ("MDD", 1)):
            group_directory = find_group_directory(site_directory, group)
            files = sorted(
                path
                for path in group_directory.rglob("*")
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            )
            valid = 0
            for path in files:
                try:
                    x, original_shape, orientation = orient_and_crop(
                        load_array(path), args.n_rois, args.timepoints
                    )
                    roi_standard_deviation = np.std(x.astype(np.float64), axis=0)
                    constant = np.flatnonzero(
                        (~np.isfinite(roi_standard_deviation))
                        | (roi_standard_deviation < args.standardization_eps)
                    )
                    if len(constant) > args.max_constant_rois:
                        raise ValueError(
                            f"{len(constant)} constant/near-constant ROIs exceeds "
                            f"--max-constant-rois={args.max_constant_rois}."
                        )
                    subject = Subject(
                        subject_id=make_subject_id(
                            args.data_root, path.resolve(), site, group
                        ),
                        path=path.resolve(),
                        site=int(site),
                        group=group,
                        label=int(label),
                        x=x,
                        content_sha256=content_hash(x),
                        original_shape=original_shape,
                        orientation=orientation,
                        n_constant_rois=int(len(constant)),
                    )
                    subjects.append(subject)
                    valid += 1
                    if len(constant):
                        constant_records.append(
                            {
                                "subject_id": subject.subject_id,
                                "path": str(subject.path),
                                "site": site,
                                "group": group,
                                "n_constant_rois": len(constant),
                                "constant_rois_1based": ";".join(
                                    str(int(index + 1)) for index in constant
                                ),
                            }
                        )
                except Exception as exc:
                    rejected.append(
                        {
                            "path": str(path),
                            "site": site,
                            "group": group,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
            print(f"[SCAN] Site {site:>2} {group:>3}: {valid}/{len(files)} valid")

    if not subjects:
        raise RuntimeError("No valid subject time series were found.")
    subjects.sort(key=lambda subject: (subject.site, subject.label, str(subject.path).lower()))

    manifest = pd.DataFrame(
        [
            {
                "subject_id": subject.subject_id,
                "path": str(subject.path),
                "site": subject.site,
                "group": subject.group,
                "label": subject.label,
                "original_rows": subject.original_shape[0],
                "original_columns": subject.original_shape[1],
                "used_timepoints": subject.x.shape[0],
                "used_rois": subject.x.shape[1],
                "orientation": subject.orientation,
                "n_constant_rois": subject.n_constant_rois,
                "content_sha256": subject.content_sha256,
            }
            for subject in subjects
        ]
    )
    manifest.to_csv(args.outdir / "dataset_manifest.csv", index=False)
    pd.DataFrame(
        rejected,
        columns=["path", "site", "group", "error_type", "error"],
    ).to_csv(args.outdir / "rejected_files.csv", index=False)
    pd.DataFrame(
        constant_records,
        columns=[
            "subject_id",
            "path",
            "site",
            "group",
            "n_constant_rois",
            "constant_rois_1based",
        ],
    ).to_csv(args.outdir / "constant_roi_qc.csv", index=False)

    duplicate_rows = []
    digest_groups: Dict[str, List[Subject]] = {}
    for subject in subjects:
        digest_groups.setdefault(subject.content_sha256, []).append(subject)
    for digest, members in digest_groups.items():
        if len(members) > 1:
            for member in members:
                duplicate_rows.append(
                    {
                        "content_sha256": digest,
                        "subject_id": member.subject_id,
                        "path": str(member.path),
                        "site": member.site,
                        "group": member.group,
                    }
                )
    pd.DataFrame(
        duplicate_rows,
        columns=["content_sha256", "subject_id", "path", "site", "group"],
    ).to_csv(args.outdir / "duplicate_time_series.csv", index=False)

    counts = pd.crosstab(
        pd.Series([subject.site for subject in subjects], name="site"),
        pd.Series([subject.group for subject in subjects], name="group"),
        margins=True,
    )
    print("\n[VALID SUBJECT COUNTS]")
    print(counts)
    print(f"\nTotal subjects: {len(subjects)}")
    if constant_records:
        affected = len({record["subject_id"] for record in constant_records})
        largest = max(record["n_constant_rois"] for record in constant_records)
        print(
            f"[QC] {affected} subject(s) contain constant/near-constant ROI channels; "
            f"the largest count is {largest}. Those channels are set to zero after "
            "safe within-subject standardization."
        )
    if duplicate_rows:
        duplicate_groups = len({row["content_sha256"] for row in duplicate_rows})
        print(
            f"[QC] WARNING: {len(duplicate_rows)} files belong to "
            f"{duplicate_groups} duplicate-content group(s)."
        )
    if rejected:
        print(f"[QC] {len(rejected)} file(s) were rejected; see rejected_files.csv.")

    missing_classes = [
        site
        for site in args.sites
        if {subject.label for subject in subjects if subject.site == site} != {0, 1}
    ]
    fatal = []
    if missing_classes:
        fatal.append(f"sites without both HC and MDD: {missing_classes}")
    if args.strict_data and rejected:
        fatal.append(f"{len(rejected)} rejected files")
    if args.strict_data and duplicate_rows:
        fatal.append("identical cropped time series detected")
    if fatal:
        raise RuntimeError(
            "Strict dataset QC failed: " + "; ".join(fatal)
            + ". Inspect the QC CSV files before using --no-strict-data."
        )
    return subjects


def safe_subject_standardize(
    x: np.ndarray, eps: float
) -> Tuple[np.ndarray, np.ndarray]:
    value = np.asarray(x, dtype=np.float64)
    mean = value.mean(axis=0, keepdims=True)
    standard_deviation = value.std(axis=0, keepdims=True)
    bad = (~np.isfinite(standard_deviation)) | (standard_deviation < eps)
    denominator = standard_deviation.copy()
    denominator[bad] = 1.0
    standardized = (value - mean) / denominator
    standardized[:, bad.ravel()] = 0.0
    if not np.all(np.isfinite(standardized)):
        raise FloatingPointError("Subject standardization produced NaN/Inf.")
    return standardized.astype(np.float32), bad.ravel()


def standardized_arrays(subjects: Sequence[Subject], eps: float) -> List[np.ndarray]:
    return [safe_subject_standardize(subject.x, eps)[0] for subject in subjects]


def fit_full_pca_rotation(
    arrays: Sequence[np.ndarray], rank_rtol: float
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    if not arrays:
        raise ValueError("No arrays were provided for PCA.")
    n_channels = arrays[0].shape[1]
    covariance = np.zeros((n_channels, n_channels), dtype=np.float64)
    n_samples = 0
    for array in arrays:
        value = np.asarray(array, dtype=np.float64)
        covariance += value.T @ value
        n_samples += value.shape[0]
    covariance /= max(n_samples - 1, 1)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    rotation = eigenvectors[:, order]
    for column in range(rotation.shape[1]):
        row = int(np.argmax(np.abs(rotation[:, column])))
        if rotation[row, column] < 0:
            rotation[:, column] *= -1.0
    threshold = max(float(eigenvalues[0]) * rank_rtol, np.finfo(float).eps)
    diagnostics = {
        "pca_components": n_channels,
        "pca_effective_rank": int(np.sum(eigenvalues > threshold)),
        "pca_rank_threshold": threshold,
        "pca_largest_eigenvalue": float(eigenvalues[0]),
        "pca_smallest_eigenvalue": float(eigenvalues[-1]),
        "pca_orthogonality_max_error": float(
            np.max(np.abs(rotation.T @ rotation - np.eye(n_channels)))
        ),
    }
    return rotation.astype(np.float32), eigenvalues, diagnostics


def build_signed_adjacency(x: np.ndarray, top_k: int) -> np.ndarray:
    value = np.asarray(x, dtype=np.float64)
    centered = value - value.mean(axis=0, keepdims=True)
    gram = centered.T @ centered
    norm = np.sqrt(np.maximum(np.diag(gram), 0.0))
    denominator = norm[:, None] * norm[None, :]
    correlation = np.zeros_like(gram)
    np.divide(gram, denominator, out=correlation, where=denominator > 1e-12)
    correlation = np.clip(correlation, -1.0, 1.0)
    np.fill_diagonal(correlation, 0.0)
    n_channels = correlation.shape[0]
    if top_k > 0 and top_k < n_channels - 1:
        mask = np.zeros_like(correlation, dtype=bool)
        for row in range(n_channels):
            indices = np.argpartition(np.abs(correlation[row]), -top_k)[-top_k:]
            mask[row, indices] = True
        mask = mask | mask.T
        correlation = np.where(mask, correlation, 0.0)
    correlation = 0.5 * (correlation + correlation.T)
    degree = np.sum(np.abs(correlation), axis=1)
    inverse_sqrt_degree = np.zeros_like(degree)
    valid = degree > 1e-12
    inverse_sqrt_degree[valid] = 1.0 / np.sqrt(degree[valid])
    normalized = (
        inverse_sqrt_degree[:, None]
        * correlation
        * inverse_sqrt_degree[None, :]
    )
    np.fill_diagonal(normalized, 0.0)
    if not np.all(np.isfinite(normalized)):
        raise FloatingPointError("Adjacency construction produced NaN/Inf.")
    return normalized.astype(np.float32)


class CiMyGnDataset(Dataset):
    def __init__(
        self,
        subjects: Sequence[Subject],
        standardization_eps: float,
        graph_top_k: int,
    ):
        self.subjects = list(subjects)
        self.arrays = standardized_arrays(self.subjects, standardization_eps)
        self.adjacencies = [
            build_signed_adjacency(array, graph_top_k) for array in self.arrays
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
            torch.from_numpy(self.adjacencies[index]),
        )


def collate_subjects(batch):
    arrays, labels, subject_ids, sites, adjacencies = zip(*batch)
    return (
        torch.stack(arrays),
        torch.stack(labels),
        list(subject_ids),
        torch.stack(sites),
        torch.stack(adjacencies),
    )


def site_class_strata(subjects: Sequence[Subject]) -> np.ndarray:
    return np.asarray(
        [f"S{subject.site}_Y{subject.label}" for subject in subjects]
    )


def split_fit_validation(
    subjects: Sequence[Subject], validation_fraction: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, str]:
    indices = np.arange(len(subjects))
    joint = site_class_strata(subjects)
    joint_counts = pd.Series(joint).value_counts()
    use_joint = bool(
        len(joint_counts)
        and np.all(joint_counts >= 2)
        and np.all(joint_counts * validation_fraction >= 1.0)
    )
    labels = np.asarray([subject.label for subject in subjects], dtype=int)
    strata = joint if use_joint else labels
    fit_index, validation_index = train_test_split(
        indices,
        test_size=validation_fraction,
        stratify=strata,
        random_state=seed,
    )
    scheme = "site-by-class" if use_joint else "class-only-fallback"
    return np.asarray(fit_index), np.asarray(validation_index), scheme
