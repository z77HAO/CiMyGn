#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ROI-aware, numerically stable CiMyGn V2 for resting-state fMRI."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def gaussian_kl(
    mean_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mean_p: torch.Tensor,
    logvar_p: torch.Tensor,
    free_bits: float = 0.0,
) -> torch.Tensor:
    """Mean Gaussian KL, optionally with free bits applied per latent unit."""
    logvar_q = torch.clamp(logvar_q, -8.0, 8.0)
    logvar_p = torch.clamp(logvar_p, -8.0, 8.0)
    var_ratio = torch.exp(logvar_q - logvar_p)
    mean_term = (mean_q - mean_p).square() * torch.exp(-logvar_p)
    elementwise = 0.5 * (var_ratio + mean_term + logvar_p - logvar_q - 1.0)
    reduce_dims = tuple(range(elementwise.ndim - 1))
    per_latent = elementwise.mean(dim=reduce_dims)
    if free_bits > 0.0:
        per_latent = torch.clamp(per_latent, min=free_bits)
    return per_latent.mean()


class GINLayer(nn.Module):
    """Weighted GIN update using a signed adjacency without self-loops."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(()))
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        messages = torch.einsum("bij,btjf->btif", adjacency, x)
        updated = (1.0 + self.eps) * x + messages
        return self.norm(self.mlp(updated))


class GraphTemporalEncoder(nn.Module):
    """ROI-aware graph encoder with direct anatomical and temporal paths."""

    def __init__(
        self,
        data_dim: int,
        gin_hidden: int,
        roi_direct_hidden: int,
        rnn_hidden: int,
        rnn_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.data_dim = int(data_dim)
        self.node_projection = nn.Linear(2, gin_hidden)
        self.roi_embedding = nn.Parameter(torch.empty(data_dim, gin_hidden))
        nn.init.normal_(self.roi_embedding, mean=0.0, std=0.02)
        self.gin1 = GINLayer(gin_hidden, gin_hidden, dropout)
        self.gin2 = GINLayer(gin_hidden, gin_hidden, dropout)
        self.node_attention = nn.Linear(gin_hidden, 1)
        self.direct_projection = nn.Sequential(
            nn.LayerNorm(2 * data_dim),
            nn.Linear(2 * data_dim, roi_direct_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.rnn = nn.LSTM(
            input_size=3 * gin_hidden + roi_direct_hidden,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )
        self.rnn_norm = nn.LayerNorm(2 * rnn_hidden)
        self.temporal_attention = nn.Linear(2 * rnn_hidden, 1)

    def forward(
        self, x: torch.Tensor, adjacency: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.shape[-1] != self.data_dim:
            raise ValueError(
                f"Expected {self.data_dim} ROIs, received {x.shape[-1]}."
            )
        node_inputs = torch.stack([x, x.square()], dim=-1)
        nodes = self.node_projection(node_inputs)
        nodes = nodes + self.roi_embedding.view(1, 1, self.data_dim, -1)
        h1 = self.gin1(nodes, adjacency)
        h2 = self.gin2(h1, adjacency) + h1

        graph_mean = h2.mean(dim=2)
        graph_max = h2.amax(dim=2)
        node_weights = torch.softmax(self.node_attention(h2).squeeze(-1), dim=2)
        graph_attention = torch.einsum("btd,btdh->bth", node_weights, h2)
        direct = self.direct_projection(torch.cat([x, x.square()], dim=-1))
        graph_sequence = torch.cat(
            [graph_mean, graph_max, graph_attention, direct], dim=-1
        )
        hidden, _ = self.rnn(graph_sequence)
        hidden = self.rnn_norm(hidden)

        temporal_weights = torch.softmax(
            self.temporal_attention(hidden).squeeze(-1), dim=1
        )
        temporal_attention = torch.einsum("bt,bth->bh", temporal_weights, hidden)
        subject_features = torch.cat(
            [
                hidden.mean(dim=1),
                hidden.std(dim=1, unbiased=False),
                temporal_attention,
            ],
            dim=-1,
        )
        return hidden, subject_features


class StaticFCEncoder(nn.Module):
    """Per-subject Fisher-z FC branch; it fits no cohort-level parameters."""

    def __init__(self, data_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        row, column = torch.triu_indices(data_dim, data_dim, offset=1)
        self.register_buffer("edge_row", row)
        self.register_buffer("edge_column", column)
        n_edges = int(row.numel())
        self.network = nn.Sequential(
            nn.LayerNorm(n_edges),
            nn.Linear(n_edges, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        centered = x - x.mean(dim=1, keepdim=True)
        norm = torch.linalg.vector_norm(centered, dim=1).clamp_min(1e-6)
        correlation = torch.einsum("btd,bte->bde", centered, centered)
        correlation = correlation / (norm.unsqueeze(-1) * norm.unsqueeze(-2))
        edges = correlation[:, self.edge_row, self.edge_column]
        fisher_z = torch.atanh(edges.clamp(-0.999, 0.999))
        return self.network(fisher_z), fisher_z


class GaussianPosterior(nn.Module):
    def __init__(self, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.to_mean = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, hidden: torch.Tensor, sample: bool):
        mean = self.to_mean(hidden)
        logvar = torch.clamp(self.to_logvar(hidden), -8.0, 8.0)
        if sample:
            latent = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        else:
            latent = mean
        return latent, mean, logvar


class ClassAutoregressivePrior(nn.Module):
    """Product of autoregressive and class-conditional Gaussian experts."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_classes: int,
        rnn_layers: int = 1,
    ):
        super().__init__()
        self.rnn = nn.LSTM(
            latent_dim,
            hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
        )
        self.to_mean = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)
        self.class_mean = nn.Parameter(torch.empty(num_classes, latent_dim))
        self.class_logvar = nn.Parameter(torch.zeros(num_classes, latent_dim))
        self.initial_h = nn.Parameter(torch.zeros(rnn_layers, 1, hidden_dim))
        self.initial_c = nn.Parameter(torch.zeros(rnn_layers, 1, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)
        with torch.no_grad():
            nn.init.normal_(self.class_mean, mean=0.0, std=0.04)
            self.class_mean.sub_(self.class_mean.mean(dim=0, keepdim=True))

    def forward(self, latent_context: torch.Tensor, labels: torch.Tensor):
        batch_size = latent_context.shape[0]
        shifted = torch.cat(
            [torch.zeros_like(latent_context[:, :1]), latent_context[:, :-1]], dim=1
        )
        h0 = self.initial_h.expand(-1, batch_size, -1).contiguous()
        c0 = self.initial_c.expand(-1, batch_size, -1).contiguous()
        ar_hidden, _ = self.rnn(shifted, (h0, c0))
        ar_hidden = self.norm(ar_hidden)
        mean_ar = self.to_mean(ar_hidden)
        logvar_ar = torch.clamp(self.to_logvar(ar_hidden), -8.0, 8.0)

        mean_class = self.class_mean[labels].unsqueeze(1).expand_as(mean_ar)
        logvar_class = (
            torch.clamp(self.class_logvar[labels], -8.0, 8.0)
            .unsqueeze(1)
            .expand_as(logvar_ar)
        )
        precision_ar = torch.exp(-logvar_ar)
        precision_class = torch.exp(-logvar_class)
        precision = precision_ar + precision_class
        variance = torch.reciprocal(precision)
        mean = variance * (precision_ar * mean_ar + precision_class * mean_class)
        return mean, torch.log(variance)


class CiMyGn(nn.Module):
    """Class-informed multi-dynamic graph network with ROI-aware fusion."""

    def __init__(self, config: Dict, pca_rotation: torch.Tensor):
        super().__init__()
        self.config = dict(config)
        self.data_dim = int(config["data_dim"])
        self.n_power_modes = int(config["n_power_modes"])
        self.n_fc_modes = int(config["n_fc_modes"])
        self.min_scale = float(config.get("min_scale", 1e-3))
        self.min_cholesky = float(config.get("min_cholesky", 1e-3))
        self.covariance_jitter = float(config.get("covariance_jitter", 1e-4))
        self.free_bits = float(config.get("free_bits", 0.02))
        self.prior_separation_margin = float(
            config.get("prior_separation_margin", 0.5)
        )
        self.coefficient_temperature = float(
            config.get("coefficient_temperature", 0.75)
        )
        self.classifier_input = str(config.get("classifier_input", "fused"))
        self.register_buffer("pca_rotation", pca_rotation.float())

        rnn_hidden = int(config["rnn_hidden"])
        self.encoder = GraphTemporalEncoder(
            data_dim=self.data_dim,
            gin_hidden=int(config["gin_hidden"]),
            roi_direct_hidden=int(config["roi_direct_hidden"]),
            rnn_hidden=rnn_hidden,
            rnn_layers=int(config["rnn_layers"]),
            dropout=float(config["dropout"]),
        )
        posterior_hidden = 2 * rnn_hidden
        self.power_posterior = GaussianPosterior(posterior_hidden, self.n_power_modes)
        self.fc_posterior = GaussianPosterior(posterior_hidden, self.n_fc_modes)
        self.power_prior = ClassAutoregressivePrior(
            self.n_power_modes, rnn_hidden, int(config["num_classes"])
        )
        self.fc_prior = ClassAutoregressivePrior(
            self.n_fc_modes, rnn_hidden, int(config["num_classes"])
        )

        scale_center = inverse_softplus(max(1.0 - self.min_scale, 1e-4))
        self.power_means = nn.Parameter(
            0.02 * torch.randn(self.n_power_modes, self.data_dim)
        )
        self.power_scale_raw = nn.Parameter(
            scale_center + 0.02 * torch.randn(self.n_power_modes, self.data_dim)
        )
        raw_tril = 0.01 * torch.tril(
            torch.randn(self.n_fc_modes, self.data_dim, self.data_dim), diagonal=-1
        )
        diagonal_raw = inverse_softplus(max(1.0 - self.min_cholesky, 1e-4))
        diagonal = torch.arange(self.data_dim)
        raw_tril[:, diagonal, diagonal] = diagonal_raw
        self.fc_cholesky_raw = nn.Parameter(raw_tril)

        dynamic_dim = (
            3 * (self.n_power_modes + self.n_fc_modes)
            + self.n_power_modes**2
            + self.n_fc_modes**2
            + 2
        )
        encoder_dim = 6 * rnn_hidden
        static_hidden = int(config["static_fc_hidden"])
        self.static_fc_encoder = StaticFCEncoder(
            self.data_dim, static_hidden, float(config["dropout"])
        )
        input_dimensions = {
            "fused": dynamic_dim + encoder_dim + static_hidden,
            "modes-only": dynamic_dim,
            "encoder-only": encoder_dim,
            "static-only": static_hidden,
        }
        if self.classifier_input not in input_dimensions:
            raise ValueError(
                f"Unknown classifier_input={self.classifier_input!r}; "
                f"choose one of {sorted(input_dimensions)}."
            )
        classifier_hidden = int(config["classifier_hidden"])
        dropout = float(config["dropout"])

        def make_classifier(input_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, classifier_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden, int(config["num_classes"])),
            )

        self.classifier = make_classifier(input_dimensions[self.classifier_input])
        self.mode_classifier = make_classifier(dynamic_dim)

    def positive_power_scales(self) -> torch.Tensor:
        return F.softplus(self.power_scale_raw) + self.min_scale

    def fc_correlation_modes(self) -> torch.Tensor:
        lower = torch.tril(self.fc_cholesky_raw, diagonal=-1)
        diagonal = F.softplus(
            torch.diagonal(self.fc_cholesky_raw, dim1=-2, dim2=-1)
        ) + self.min_cholesky
        cholesky = lower + torch.diag_embed(diagonal)
        covariance = cholesky @ cholesky.transpose(-1, -2)
        standard_deviation = torch.sqrt(
            torch.diagonal(covariance, dim1=-2, dim2=-1).clamp_min(1e-8)
        )
        correlation = covariance / (
            standard_deviation.unsqueeze(-1) * standard_deviation.unsqueeze(-2)
        )
        return 0.5 * (correlation + correlation.transpose(-1, -2))

    @staticmethod
    def temporal_summary(coefficients: torch.Tensor) -> torch.Tensor:
        mean = coefficients.mean(dim=1)
        standard_deviation = coefficients.std(dim=1, unbiased=False)
        if coefficients.shape[1] > 1:
            mean_absolute_difference = coefficients.diff(dim=1).abs().mean(dim=1)
        else:
            mean_absolute_difference = torch.zeros_like(mean)
        return torch.cat([mean, standard_deviation, mean_absolute_difference], dim=-1)

    @staticmethod
    def transition_summary(coefficients: torch.Tensor) -> torch.Tensor:
        batch_size, n_timepoints, n_modes = coefficients.shape
        if n_timepoints <= 1:
            return coefficients.new_zeros((batch_size, n_modes * n_modes))
        transition = torch.einsum(
            "bti,btj->bij", coefficients[:, :-1], coefficients[:, 1:]
        ) / (n_timepoints - 1)
        return transition.reshape(batch_size, n_modes * n_modes)

    def classification_features(
        self, power_coefficients: torch.Tensor, fc_coefficients: torch.Tensor
    ) -> torch.Tensor:
        eps = 1e-8
        power_entropy = -(
            power_coefficients.clamp_min(eps)
            * power_coefficients.clamp_min(eps).log()
        ).sum(dim=-1).mean(dim=1, keepdim=True)
        fc_entropy = -(
            fc_coefficients.clamp_min(eps) * fc_coefficients.clamp_min(eps).log()
        ).sum(dim=-1).mean(dim=1, keepdim=True)
        return torch.cat(
            [
                self.temporal_summary(power_coefficients),
                self.temporal_summary(fc_coefficients),
                self.transition_summary(power_coefficients),
                self.transition_summary(fc_coefficients),
                power_entropy,
                fc_entropy,
            ],
            dim=-1,
        )

    @staticmethod
    def _cosine_diversity_penalty(signatures: torch.Tensor) -> torch.Tensor:
        if signatures.shape[0] <= 1:
            return signatures.new_zeros(())
        signatures = signatures - signatures.mean(dim=-1, keepdim=True)
        signatures = F.normalize(signatures, dim=-1, eps=1e-6)
        gram = signatures @ signatures.transpose(-1, -2)
        mask = ~torch.eye(
            signatures.shape[0], dtype=torch.bool, device=signatures.device
        )
        return gram[mask].square().mean()

    def mode_diversity_penalty(self) -> torch.Tensor:
        power_signature = torch.cat(
            [self.power_means, torch.log(self.positive_power_scales())], dim=-1
        )
        correlation = self.fc_correlation_modes()
        row, column = torch.triu_indices(
            self.data_dim, self.data_dim, offset=1, device=correlation.device
        )
        fc_signature = correlation[:, row, column]
        return self._cosine_diversity_penalty(
            power_signature
        ) + self._cosine_diversity_penalty(fc_signature)

    def prior_separation_penalty(self) -> torch.Tensor:
        penalties = []
        for prior in (self.power_prior, self.fc_prior):
            distances = torch.pdist(prior.class_mean, p=2)
            if distances.numel():
                penalties.append(
                    F.relu(self.prior_separation_margin - distances).mean()
                )
        if not penalties:
            return self.power_means.new_zeros(())
        return torch.stack(penalties).mean()

    @staticmethod
    def occupancy_regularizers(
        power: torch.Tensor, fc: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        entropies = []
        balance = []
        for coefficients in (power, fc):
            n_modes = coefficients.shape[-1]
            entropy = -(
                coefficients.clamp_min(1e-8)
                * coefficients.clamp_min(1e-8).log()
            ).sum(dim=-1)
            entropies.append(entropy.mean() / math.log(n_modes))
            average = coefficients.mean(dim=(0, 1))
            target = torch.full_like(average, 1.0 / n_modes)
            balance.append(n_modes * (average - target).square().sum())
        return torch.stack(entropies).mean(), torch.stack(balance).mean()

    def gaussian_nll(
        self,
        x: torch.Tensor,
        power_coefficients: torch.Tensor,
        fc_coefficients: torch.Tensor,
        n_time_samples: int,
    ) -> torch.Tensor:
        _, n_timepoints, n_channels = x.shape
        if n_time_samples > 0 and n_time_samples < n_timepoints:
            time_index = torch.randperm(n_timepoints, device=x.device)[:n_time_samples]
            x = x.index_select(1, time_index)
            power_coefficients = power_coefficients.index_select(1, time_index)
            fc_coefficients = fc_coefficients.index_select(1, time_index)

        x_pca = x @ self.pca_rotation
        mean = torch.einsum("btk,kd->btd", power_coefficients, self.power_means)
        scale = torch.einsum(
            "btk,kd->btd", power_coefficients, self.positive_power_scales()
        )
        correlation = torch.einsum(
            "btq,qij->btij", fc_coefficients, self.fc_correlation_modes()
        )
        covariance = scale.unsqueeze(-1) * correlation * scale.unsqueeze(-2)
        identity = torch.eye(n_channels, device=x.device, dtype=x.dtype)

        jitter = self.covariance_jitter
        cholesky = None
        last_info = None
        for _ in range(6):
            candidate, info = torch.linalg.cholesky_ex(
                covariance + jitter * identity
            )
            if bool(torch.all(info == 0)):
                cholesky = candidate
                break
            last_info = info
            jitter *= 10.0
        if cholesky is None:
            failed = int((last_info != 0).sum().item()) if last_info is not None else -1
            raise RuntimeError(
                f"Cholesky decomposition failed for {failed} covariance matrices "
                "after bounded jitter escalation."
            )

        residual = (x_pca - mean).unsqueeze(-1)
        solution = torch.cholesky_solve(residual, cholesky)
        mahalanobis = (residual.transpose(-1, -2) @ solution).squeeze(-1).squeeze(-1)
        log_determinant = 2.0 * torch.log(
            torch.diagonal(cholesky, dim1=-2, dim2=-1)
        ).sum(dim=-1)
        nll = 0.5 * (
            mahalanobis + log_determinant + n_channels * math.log(2.0 * math.pi)
        )
        return (nll / n_channels).mean()

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        sample_latent: Optional[bool] = None,
        compute_nll: bool = True,
        nll_time_samples: int = 0,
        class_weights: Optional[torch.Tensor] = None,
        compute_regularizers: Optional[bool] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if sample_latent is None:
            sample_latent = self.training
        if compute_regularizers is None:
            compute_regularizers = labels is not None
        hidden, encoder_subject_features = self.encoder(x, adjacency)
        static_fc_embedding, _ = self.static_fc_encoder(x)
        power_z, power_mean, power_logvar = self.power_posterior(
            hidden, sample=sample_latent
        )
        fc_z, fc_mean, fc_logvar = self.fc_posterior(hidden, sample=sample_latent)

        power_for_nll = F.softmax(power_z / self.coefficient_temperature, dim=-1)
        fc_for_nll = F.softmax(fc_z / self.coefficient_temperature, dim=-1)
        power_coefficients = F.softmax(
            power_mean / self.coefficient_temperature, dim=-1
        )
        fc_coefficients = F.softmax(fc_mean / self.coefficient_temperature, dim=-1)
        dynamic_features = self.classification_features(
            power_coefficients, fc_coefficients
        )
        feature_options = {
            "fused": torch.cat(
                [dynamic_features, encoder_subject_features, static_fc_embedding], dim=-1
            ),
            "modes-only": dynamic_features,
            "encoder-only": encoder_subject_features,
            "static-only": static_fc_embedding,
        }
        logits = self.classifier(feature_options[self.classifier_input])
        mode_logits = self.mode_classifier(dynamic_features)

        nll = (
            self.gaussian_nll(
                x, power_for_nll, fc_for_nll, n_time_samples=nll_time_samples
            )
            if compute_nll
            else None
        )
        kl = None
        kl_raw = None
        classification_loss = None
        mode_classification_loss = None
        if labels is not None:
            power_prior_mean, power_prior_logvar = self.power_prior(
                power_z.detach(), labels
            )
            fc_prior_mean, fc_prior_logvar = self.fc_prior(fc_z.detach(), labels)
            kl_raw = gaussian_kl(
                power_mean, power_logvar, power_prior_mean, power_prior_logvar
            ) + gaussian_kl(fc_mean, fc_logvar, fc_prior_mean, fc_prior_logvar)
            kl = gaussian_kl(
                power_mean,
                power_logvar,
                power_prior_mean,
                power_prior_logvar,
                self.free_bits,
            ) + gaussian_kl(
                fc_mean,
                fc_logvar,
                fc_prior_mean,
                fc_prior_logvar,
                self.free_bits,
            )
            classification_loss = F.cross_entropy(
                logits, labels, weight=class_weights
            )
            mode_classification_loss = F.cross_entropy(
                mode_logits, labels, weight=class_weights
            )

        if compute_regularizers:
            prior_separation = self.prior_separation_penalty()
            mode_diversity = self.mode_diversity_penalty()
            mode_entropy, mode_balance = self.occupancy_regularizers(
                power_coefficients, fc_coefficients
            )
        else:
            prior_separation = None
            mode_diversity = None
            mode_entropy = None
            mode_balance = None
        return {
            "logits": logits,
            "mode_logits": mode_logits,
            "nll": nll,
            "kl": kl,
            "kl_raw": kl_raw,
            "classification_loss": classification_loss,
            "mode_classification_loss": mode_classification_loss,
            "prior_separation": prior_separation,
            "mode_diversity": mode_diversity,
            "mode_entropy": mode_entropy,
            "mode_balance": mode_balance,
            "power_coefficients": power_coefficients,
            "fc_coefficients": fc_coefficients,
            "power_mean": power_mean,
            "fc_mean": fc_mean,
            "classifier_features": dynamic_features,
            "encoder_subject_features": encoder_subject_features,
            "static_fc_embedding": static_fc_embedding,
        }

    @staticmethod
    def total_loss(
        outputs: Dict[str, Optional[torch.Tensor]],
        epoch: int,
        nll_weight: float,
        kl_max_weight: float,
        classification_weight: float,
        mode_classification_weight: float,
        classification_warmup_epochs: int,
        generative_ramp_epochs: int,
        prior_separation_weight: float,
        mode_diversity_weight: float,
        occupancy_entropy_weight: float,
        occupancy_balance_weight: float,
    ):
        required = (
            "classification_loss",
            "mode_classification_loss",
            "prior_separation",
            "mode_diversity",
            "mode_entropy",
            "mode_balance",
        )
        missing = [name for name in required if outputs[name] is None]
        if missing:
            raise ValueError(
                "Training loss requires labels and regularizers; missing "
                + ", ".join(missing)
            )
        if epoch <= classification_warmup_epochs:
            generative_fraction = 0.0
        else:
            generative_fraction = min(
                1.0,
                (epoch - classification_warmup_epochs)
                / max(generative_ramp_epochs, 1),
            )
        effective_nll = nll_weight * generative_fraction
        effective_kl = kl_max_weight * generative_fraction
        if effective_nll > 0.0 and outputs["nll"] is None:
            raise ValueError("A positive effective NLL weight requires an NLL value.")
        if effective_kl > 0.0 and outputs["kl"] is None:
            raise ValueError("A positive effective KL weight requires a KL value.")

        zero = outputs["classification_loss"].new_zeros(())
        total = (
            classification_weight * outputs["classification_loss"]
            + mode_classification_weight * outputs["mode_classification_loss"]
            + effective_nll * (outputs["nll"] if outputs["nll"] is not None else zero)
            + effective_kl * (outputs["kl"] if outputs["kl"] is not None else zero)
            + prior_separation_weight
            * generative_fraction
            * outputs["prior_separation"]
            + mode_diversity_weight * outputs["mode_diversity"]
            + occupancy_entropy_weight * outputs["mode_entropy"]
            + occupancy_balance_weight * outputs["mode_balance"]
        )
        weights = {
            "generative_fraction": generative_fraction,
            "nll_weight": effective_nll,
            "kl_weight": effective_kl,
            "prior_separation_weight": prior_separation_weight * generative_fraction,
        }
        return total, weights
