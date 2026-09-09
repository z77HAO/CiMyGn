#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Small CPU self-test for the manuscript-aligned CiMyGn model."""

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
        "num_classes": 2,
        "encoder_hidden": 8,
        "encoder_layers": 2,
        "encoder_dropout": 0.0,
        "prior_hidden": 8,
        "classifier_hidden": 12,
        "classifier_dropout": 0.0,
        "tau_power": 1.0,
        "tau_fc": 1.0,
        "min_scale": 1e-4,
        "min_cholesky": 1e-4,
        "covariance_jitter": 1e-5,
    }
    rotation, _ = torch.linalg.qr(torch.randn(data_dim, data_dim))
    model = CiMyGn(config, rotation)
    x = torch.randn(batch_size, timepoints, data_dim)
    labels = torch.tensor([0, 1, 1])

    model.train()
    outputs = model(
        x,
        labels=labels,
        sample_latent=True,
        compute_nll=True,
        nll_chunk_size=4,
    )
    loss = model.total_loss(outputs, lambda_kl=0.5, gamma_cls=1.0)
    if not torch.isfinite(loss):
        raise AssertionError("Training loss is not finite.")
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
        raise AssertionError("A model gradient is missing or non-finite.")

    power_scale = model.power_scale_templates().detach()
    if float(power_scale.min()) <= 0.0:
        raise AssertionError("Regional-scale templates are not strictly positive.")

    correlations = model.fc_correlation_templates().detach()
    if not torch.allclose(
        correlations, correlations.transpose(-1, -2), atol=1e-6
    ):
        raise AssertionError("FC templates are not symmetric.")
    diagonal = torch.diagonal(correlations, dim1=-2, dim2=-1)
    if not torch.allclose(diagonal, torch.ones_like(diagonal), atol=1e-5):
        raise AssertionError("FC templates do not have unit diagonal.")
    if float(torch.linalg.eigvalsh(correlations).min()) <= 0.0:
        raise AssertionError("An FC template is not positive definite.")

    model.eval()
    with torch.no_grad():
        first = model(x, labels=None, sample_latent=False, compute_nll=False)
        second = model(x, labels=None, sample_latent=False, compute_nll=False)

    if first["kl"] is not None or first["classification_loss"] is not None:
        raise AssertionError("Unlabelled inference computed labelled losses.")
    if not torch.equal(first["logits"], second["logits"]):
        raise AssertionError("Deterministic inference produced different logits.")
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
    if first["prior_power_mean"] is not None or first["prior_fc_mean"] is not None:
        raise AssertionError("Class-guided prior was evaluated without labels.")

    print("CiMyGn manuscript-aligned self-test: PASS")
    print(f"torch={torch.__version__}; loss={float(loss.detach()):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
