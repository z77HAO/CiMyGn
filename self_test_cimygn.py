#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Small CPU self-test for the CiMyGn model; no study data are required."""

from __future__ import annotations

import torch

from cimygn_model import CiMyGn


def main() -> int:
    torch.manual_seed(7)
    batch_size, timepoints, data_dim = 3, 12, 8
    config = {
        "data_dim": data_dim,
        "n_power_modes": 3,
        "n_fc_modes": 4,
        "gin_hidden": 6,
        "roi_direct_hidden": 7,
        "rnn_hidden": 8,
        "rnn_layers": 1,
        "static_fc_hidden": 9,
        "classifier_hidden": 12,
        "classifier_input": "fused",
        "dropout": 0.0,
        "num_classes": 2,
        "min_scale": 1e-3,
        "min_cholesky": 1e-3,
        "covariance_jitter": 1e-4,
        "free_bits": 0.02,
        "prior_separation_margin": 0.5,
        "coefficient_temperature": 0.75,
    }
    orthogonal, _ = torch.linalg.qr(torch.randn(data_dim, data_dim))
    model = CiMyGn(config, orthogonal)
    x = torch.randn(batch_size, timepoints, data_dim)
    adjacency = torch.randn(batch_size, data_dim, data_dim)
    adjacency = 0.5 * (adjacency + adjacency.transpose(-1, -2))
    adjacency.diagonal(dim1=-2, dim2=-1).zero_()
    degree = adjacency.abs().sum(dim=-1).clamp_min(1e-6)
    adjacency = (
        degree.rsqrt().unsqueeze(-1)
        * adjacency
        * degree.rsqrt().unsqueeze(-2)
    )
    labels = torch.tensor([0, 1, 1])

    model.train()
    outputs = model(
        x,
        adjacency,
        labels=labels,
        sample_latent=True,
        compute_nll=True,
        nll_time_samples=6,
    )
    loss, weights = model.total_loss(
        outputs,
        epoch=3,
        nll_weight=0.25,
        kl_max_weight=0.02,
        classification_weight=1.0,
        mode_classification_weight=0.25,
        classification_warmup_epochs=1,
        generative_ramp_epochs=2,
        prior_separation_weight=0.01,
        mode_diversity_weight=0.01,
        occupancy_entropy_weight=0.002,
        occupancy_balance_weight=0.01,
    )
    if not torch.isfinite(loss):
        raise AssertionError("Training loss is not finite.")
    loss.backward()
    finite_gradients = [
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not finite_gradients or not all(bool(value) for value in finite_gradients):
        raise AssertionError("A model gradient is missing or non-finite.")
    if weights["generative_fraction"] != 1.0:
        raise AssertionError("The staged loss schedule did not reach full weight.")

    correlations = model.fc_correlation_modes().detach()
    if not torch.allclose(
        correlations, correlations.transpose(-1, -2), atol=1e-6
    ):
        raise AssertionError("FC modes are not symmetric.")
    if float(torch.linalg.eigvalsh(correlations).min()) <= 0.0:
        raise AssertionError("An FC correlation mode is not positive definite.")

    model.eval()
    with torch.no_grad():
        first = model(
            x,
            adjacency,
            labels=None,
            sample_latent=False,
            compute_nll=False,
        )
        second = model(
            x,
            adjacency,
            labels=None,
            sample_latent=False,
            compute_nll=False,
        )
    if first["kl"] is not None or first["classification_loss"] is not None:
        raise AssertionError("Unlabelled inference unexpectedly computed labelled losses.")
    if not torch.equal(first["logits"], second["logits"]):
        raise AssertionError("Deterministic evaluation produced different logits.")
    if not torch.allclose(
        first["power_coefficients"].sum(dim=-1),
        torch.ones(batch_size, timepoints),
        atol=1e-6,
    ):
        raise AssertionError("Power coefficients do not sum to one.")
    if not torch.allclose(
        first["fc_coefficients"].sum(dim=-1),
        torch.ones(batch_size, timepoints),
        atol=1e-6,
    ):
        raise AssertionError("FC coefficients do not sum to one.")
    if first["encoder_subject_features"].shape != (batch_size, 48):
        raise AssertionError("Unexpected ROI-aware encoder feature shape.")
    if first["static_fc_embedding"].shape != (batch_size, 9):
        raise AssertionError("Unexpected static-FC embedding shape.")

    print("CiMyGn self-test: PASS")
    print(f"torch={torch.__version__}; loss={float(loss.detach()):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
