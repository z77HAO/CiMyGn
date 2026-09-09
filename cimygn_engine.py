#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CiMyGn model aligned with the method described in the manuscript.

The implementation follows the manuscript at the level explicitly specified there:
- a two-layer BiLSTM inference encoder;
- separate diagonal-Gaussian posterior heads for regional-scale and FC logits;
- one class-guided recurrent prior with a shared recurrent state and separate
  output parameterizations for the two branches;
- temperature-softmax mode-expression weights;
- positive diagonal regional-scale templates and SPD/unit-diagonal FC templates;
- zero-mean Gaussian observation model with C_t = G_t F_t G_t;
- a subject-level MLP classifier derived from the inferred latent trajectories;
- training loss: NLL + lambda_KL * KL + gamma_cls * CE.

The manuscript does not specify every low-level engineering choice (for example,
the exact KL-annealing schedule or tensor-reduction convention). Those choices are
implemented in cimygn_engine.py and are kept explicit there.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-8)
    return math.log(math.expm1(value))


def diagonal_gaussian_kl(
    mean_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mean_p: torch.Tensor,
    logvar_p: torch.Tensor,
) -> torch.Tensor:
    """KL[q||p] for diagonal Gaussians, summed over latent dimensions.

    Returns a tensor with the latent dimension removed, e.g. [B, T].
    """
    logvar_q = torch.clamp(logvar_q, -12.0, 12.0)
    logvar_p = torch.clamp(logvar_p, -12.0, 12.0)
    variance_ratio = torch.exp(logvar_q - logvar_p)
    mean_term = (mean_q - mean_p).square() * torch.exp(-logvar_p)
    elementwise = 0.5 * (
        variance_ratio + mean_term + logvar_p - logvar_q - 1.0
    )
    return elementwise.sum(dim=-1)


class BiLSTMInferenceEncoder(nn.Module):
    """Bidirectional LSTM inference encoder q_psi(Theta_1:T | x_1:T)."""

    def __init__(
        self,
        data_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.data_dim = int(data_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.rnn = nn.LSTM(
            input_size=self.data_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=float(dropout) if self.num_layers > 1 else 0.0,
        )
        self.output_norm = nn.LayerNorm(2 * self.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.data_dim:
            raise ValueError(
                f"Expected x with shape [B,T,{self.data_dim}], received {tuple(x.shape)}."
            )
        hidden, _ = self.rnn(x)
        return self.output_norm(hidden)


class GaussianPosteriorHead(nn.Module):
    """Diagonal-Gaussian posterior head for one latent branch."""

    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.mean = nn.Linear(input_dim, latent_dim)
        self.logvar = nn.Linear(input_dim, latent_dim)

    def forward(
        self, hidden: torch.Tensor, sample: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.mean(hidden)
        logvar = torch.clamp(self.logvar(hidden), -10.0, 10.0)
        if sample:
            eps = torch.randn_like(mean)
            latent = mean + torch.exp(0.5 * logvar) * eps
        else:
            latent = mean
        return latent, mean, logvar


class ClassGuidedSharedRecurrentPrior(nn.Module):
    """Diagnosis-conditioned recurrent prior with one shared recurrent state.

    The manuscript states that the power and FC branches retain separate prior
    parameterizations while being coupled through a shared recurrent state. This
    implementation realizes that statement by feeding the shifted joint latent
    history and one-hot class label into one unidirectional LSTM, followed by
    branch-specific Gaussian output heads.
    """

    def __init__(
        self,
        power_dim: int,
        fc_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.power_dim = int(power_dim)
        self.fc_dim = int(fc_dim)
        self.joint_dim = self.power_dim + self.fc_dim
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)

        self.rnn = nn.LSTM(
            input_size=self.joint_dim + self.num_classes,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(self.hidden_dim)

        self.power_mean = nn.Linear(self.hidden_dim, self.power_dim)
        self.power_logvar = nn.Linear(self.hidden_dim, self.power_dim)
        self.fc_mean = nn.Linear(self.hidden_dim, self.fc_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, self.fc_dim)

    def forward(
        self,
        sampled_power: torch.Tensor,
        sampled_fc: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if labels is None:
            raise ValueError("The class-guided prior requires labels during training.")
        joint = torch.cat([sampled_power, sampled_fc], dim=-1)
        shifted = torch.cat([torch.zeros_like(joint[:, :1]), joint[:, :-1]], dim=1)
        one_hot = F.one_hot(labels.long(), num_classes=self.num_classes).to(joint.dtype)
        one_hot = one_hot.unsqueeze(1).expand(-1, joint.shape[1], -1)
        prior_input = torch.cat([shifted, one_hot], dim=-1)
        recurrent, _ = self.rnn(prior_input)
        recurrent = self.norm(recurrent)

        power_mean = self.power_mean(recurrent)
        power_logvar = torch.clamp(self.power_logvar(recurrent), -10.0, 10.0)
        fc_mean = self.fc_mean(recurrent)
        fc_logvar = torch.clamp(self.fc_logvar(recurrent), -10.0, 10.0)
        return power_mean, power_logvar, fc_mean, fc_logvar


class CiMyGn(nn.Module):
    """Class-Guided Multi-dynamics Generative network."""

    def __init__(self, config: Dict, pca_rotation: torch.Tensor) -> None:
        super().__init__()
        self.config = dict(config)
        self.data_dim = int(config["data_dim"])
        self.n_power_modes = int(config.get("n_power_modes", 3))
        self.n_fc_modes = int(config.get("n_fc_modes", 6))
        self.num_classes = int(config.get("num_classes", 2))

        encoder_hidden = int(config.get("encoder_hidden", 128))
        encoder_layers = int(config.get("encoder_layers", 2))
        encoder_dropout = float(config.get("encoder_dropout", 0.2))
        prior_hidden = int(config.get("prior_hidden", 128))
        classifier_hidden = int(config.get("classifier_hidden", 256))
        classifier_dropout = float(config.get("classifier_dropout", 0.5))
        self.tau_power = float(config.get("tau_power", 1.0))
        self.tau_fc = float(config.get("tau_fc", 1.0))
        self.min_scale = float(config.get("min_scale", 1e-4))
        self.min_cholesky = float(config.get("min_cholesky", 1e-4))
        self.covariance_jitter = float(config.get("covariance_jitter", 1e-5))

        if self.tau_power <= 0.0 or self.tau_fc <= 0.0:
            raise ValueError("Softmax temperatures must be positive.")

        rotation = torch.as_tensor(pca_rotation, dtype=torch.float32)
        if rotation.shape != (self.data_dim, self.data_dim):
            raise ValueError(
                "pca_rotation must be square with shape "
                f"({self.data_dim},{self.data_dim}); received {tuple(rotation.shape)}."
            )
        self.register_buffer("pca_rotation", rotation)

        self.encoder = BiLSTMInferenceEncoder(
            data_dim=self.data_dim,
            hidden_dim=encoder_hidden,
            num_layers=encoder_layers,
            dropout=encoder_dropout,
        )
        posterior_input = 2 * encoder_hidden
        self.power_posterior = GaussianPosteriorHead(
            posterior_input, self.n_power_modes
        )
        self.fc_posterior = GaussianPosteriorHead(posterior_input, self.n_fc_modes)

        self.prior = ClassGuidedSharedRecurrentPrior(
            power_dim=self.n_power_modes,
            fc_dim=self.n_fc_modes,
            hidden_dim=prior_hidden,
            num_classes=self.num_classes,
        )

        # E_k: positive diagonal regional-scale templates (standard deviations).
        init_scale = _inverse_softplus(1.0 - self.min_scale)
        self.power_scale_raw = nn.Parameter(
            init_scale
            + 0.02 * torch.randn(self.n_power_modes, self.data_dim)
        )

        # R_q: SPD, unit-diagonal FC templates via Cholesky + normalization.
        raw = 0.01 * torch.tril(
            torch.randn(self.n_fc_modes, self.data_dim, self.data_dim), diagonal=-1
        )
        diag_raw = _inverse_softplus(1.0 - self.min_cholesky)
        diagonal = torch.arange(self.data_dim)
        raw[:, diagonal, diagonal] = diag_raw
        self.fc_cholesky_raw = nn.Parameter(raw)

        # Subject-level representation derived from inferred power/FC trajectories.
        # The manuscript does not specify the pooling operator; temporal mean pooling
        # is used here as the minimal parameter-free realization.
        classifier_input_dim = self.n_power_modes + self.n_fc_modes
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Linear(classifier_input_dim, classifier_hidden),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden, self.num_classes),
        )

    def power_scale_templates(self) -> torch.Tensor:
        """Return diag(E_k), i.e. positive regional standard-deviation templates."""
        return F.softplus(self.power_scale_raw) + self.min_scale

    def power_variance_templates(self) -> torch.Tensor:
        """Return v_k = diag(E_k)^2 from manuscript Eq. (13)."""
        return self.power_scale_templates().square()

    def fc_correlation_templates(self) -> torch.Tensor:
        """Return SPD unit-diagonal FC templates R_q."""
        lower = torch.tril(self.fc_cholesky_raw, diagonal=-1)
        diagonal = F.softplus(
            torch.diagonal(self.fc_cholesky_raw, dim1=-2, dim2=-1)
        ) + self.min_cholesky
        cholesky = lower + torch.diag_embed(diagonal)
        covariance = cholesky @ cholesky.transpose(-1, -2)
        sd = torch.sqrt(
            torch.diagonal(covariance, dim1=-2, dim2=-1).clamp_min(1e-12)
        )
        correlation = covariance / (sd.unsqueeze(-1) * sd.unsqueeze(-2))
        correlation = 0.5 * (correlation + correlation.transpose(-1, -2))
        return correlation

    @staticmethod
    def _trajectory_summary(
        power_mean: torch.Tensor, fc_mean: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat(
            [power_mean.mean(dim=1), fc_mean.mean(dim=1)],
            dim=-1,
        )

    def mode_weights(
        self, power_logits: torch.Tensor, fc_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha = F.softmax(power_logits / self.tau_power, dim=-1)
        beta = F.softmax(fc_logits / self.tau_fc, dim=-1)
        return alpha, beta

    def _negative_log_likelihood(
        self,
        x: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        chunk_size: int = 16,
    ) -> torch.Tensor:
        """Zero-mean Gaussian NLL under C_t = G_t F_t G_t.

        All time points are used. ``chunk_size`` only controls memory use; it does
        not subsample the sequence. The returned value is averaged over subjects,
        time points, and ROIs for stable optimization.
        """
        batch_size, n_timepoints, n_channels = x.shape
        if n_channels != self.data_dim:
            raise ValueError("Unexpected ROI dimension in likelihood evaluation.")
        chunk_size = n_timepoints if chunk_size <= 0 else int(chunk_size)

        scale_templates = self.power_scale_templates()
        fc_templates = self.fc_correlation_templates()
        rotation = self.pca_rotation.to(dtype=x.dtype)
        identity = torch.eye(n_channels, dtype=x.dtype, device=x.device)
        constant = n_channels * math.log(2.0 * math.pi)

        total = x.new_zeros(())
        count = 0
        for start in range(0, n_timepoints, chunk_size):
            stop = min(start + chunk_size, n_timepoints)
            x_chunk = x[:, start:stop]
            alpha_chunk = alpha[:, start:stop]
            beta_chunk = beta[:, start:stop]

            g = torch.einsum("btk,kd->btd", alpha_chunk, scale_templates)
            f = torch.einsum("btq,qde->btde", beta_chunk, fc_templates)
            covariance = g.unsqueeze(-1) * f * g.unsqueeze(-2)

            # Fixed full-rank PCA rotation used only for Gaussian-likelihood
            # evaluation: x' = xR and C' = R^T C R.
            x_rot = x_chunk @ rotation
            covariance_rot = torch.einsum(
                "di,btde,ej->btij", rotation, covariance, rotation
            )
            covariance_rot = 0.5 * (
                covariance_rot + covariance_rot.transpose(-1, -2)
            )

            jitter = self.covariance_jitter
            cholesky = None
            info = None
            for _ in range(7):
                candidate, info = torch.linalg.cholesky_ex(
                    covariance_rot + jitter * identity
                )
                if bool(torch.all(info == 0)):
                    cholesky = candidate
                    break
                jitter *= 10.0
            if cholesky is None:
                failed = int((info != 0).sum().item()) if info is not None else -1
                raise RuntimeError(
                    f"Cholesky decomposition failed for {failed} covariance matrices."
                )

            residual = x_rot.unsqueeze(-1)
            solved = torch.cholesky_solve(residual, cholesky)
            mahalanobis = (
                residual.transpose(-1, -2) @ solved
            ).squeeze(-1).squeeze(-1)
            logdet = 2.0 * torch.log(
                torch.diagonal(cholesky, dim1=-2, dim2=-1)
            ).sum(dim=-1)
            nll = 0.5 * (mahalanobis + logdet + constant)
            total = total + nll.sum()
            count += batch_size * (stop - start)

        return total / max(count * n_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        sample_latent: Optional[bool] = None,
        compute_nll: bool = True,
        nll_chunk_size: int = 16,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if sample_latent is None:
            sample_latent = self.training

        hidden = self.encoder(x)
        power_z, power_mean, power_logvar = self.power_posterior(
            hidden, sample=sample_latent
        )
        fc_z, fc_mean, fc_logvar = self.fc_posterior(
            hidden, sample=sample_latent
        )

        # Sampled logits are used for the Monte-Carlo generative path.
        alpha_sample, beta_sample = self.mode_weights(power_z, fc_z)
        # Posterior means give deterministic trajectories for prediction/reporting.
        alpha_mean, beta_mean = self.mode_weights(power_mean, fc_mean)

        subject_representation = self._trajectory_summary(power_mean, fc_mean)
        logits = self.classifier(subject_representation)

        nll = (
            self._negative_log_likelihood(
                x, alpha_sample, beta_sample, chunk_size=nll_chunk_size
            )
            if compute_nll
            else None
        )

        kl = None
        classification_loss = None
        prior_outputs = (None, None, None, None)
        if labels is not None:
            prior_outputs = self.prior(power_z, fc_z, labels)
            p_power_mean, p_power_logvar, p_fc_mean, p_fc_logvar = prior_outputs
            kl_power = diagonal_gaussian_kl(
                power_mean, power_logvar, p_power_mean, p_power_logvar
            )
            kl_fc = diagonal_gaussian_kl(
                fc_mean, fc_logvar, p_fc_mean, p_fc_logvar
            )
            # Mean over subjects/time and normalize by joint latent dimensionality.
            kl = (kl_power + kl_fc).mean() / (
                self.n_power_modes + self.n_fc_modes
            )
            classification_loss = F.cross_entropy(logits, labels.long())

        return {
            "logits": logits,
            "nll": nll,
            "kl": kl,
            "classification_loss": classification_loss,
            "power_sample": power_z,
            "fc_sample": fc_z,
            "power_mean": power_mean,
            "power_logvar": power_logvar,
            "fc_mean": fc_mean,
            "fc_logvar": fc_logvar,
            "power_coefficients": alpha_mean,
            "fc_coefficients": beta_mean,
            "power_coefficients_sampled": alpha_sample,
            "fc_coefficients_sampled": beta_sample,
            "subject_representation": subject_representation,
            "prior_power_mean": prior_outputs[0],
            "prior_power_logvar": prior_outputs[1],
            "prior_fc_mean": prior_outputs[2],
            "prior_fc_logvar": prior_outputs[3],
        }

    @staticmethod
    def total_loss(
        outputs: Dict[str, Optional[torch.Tensor]],
        lambda_kl: float,
        gamma_cls: float,
    ) -> torch.Tensor:
        """Manuscript Eq. (21): -L_ll + lambda_kl L_kl + gamma_cls L_cls."""
        if outputs["nll"] is None:
            raise ValueError("Training loss requires the Gaussian NLL.")
        if outputs["kl"] is None or outputs["classification_loss"] is None:
            raise ValueError("Training loss requires labels for KL and classification.")
        return (
            outputs["nll"]
            + float(lambda_kl) * outputs["kl"]
            + float(gamma_cls) * outputs["classification_loss"]
        )
