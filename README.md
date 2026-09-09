# CiMyGn

## Class-Guided Generative Modeling of Brain Dynamics for Major Depressive Disorder Diagnosis from rs-fMRI

This repository provides the implementation of the **Class-guided Multi-dynamics Generative network (CiMyGn)** for modeling diagnosis-associated brain dynamics from resting-state functional magnetic resonance imaging (rs-fMRI).

CiMyGn models regional fluctuation magnitude and functional connectivity (FC) as two distinct but coupled latent dynamical processes. A class-guided recurrent prior introduces diagnosis-conditioned temporal regularization of the coupled power and FC trajectories, while an end-to-end diagnostic classifier provides direct supervision for major depressive disorder (MDD) classification.

In this framework, **regional power** refers to the model-implied conditional variance of standardized regional BOLD fluctuations. It should not be interpreted as mean BOLD intensity or frequency-domain spectral power.

---

## Method Overview

Let

\[
x_{1:T}=\{x_t\}_{t=1}^{T}, \qquad x_t\in\mathbb{R}^{D},
\]

denote an rs-fMRI sequence, where \(D\) is the number of regions of interest (ROIs) and \(T\) is the sequence length.

CiMyGn contains two coupled latent branches representing regional power and functional connectivity.

A bidirectional long short-term memory (BiLSTM) encoder maps the observed rs-fMRI sequence to temporally contextualized representations. Separate variational posterior heads infer latent-logit trajectories for the power and FC branches.

For each branch \(b\in\{P,FC\}\),

\[
q_{\psi}(\theta_t^b\mid x_{1:T})
=
\mathcal{N}
\left(
\theta_t^b;
m_t^b,
\operatorname{Diag}\left[(s_t^b)^2\right]
\right).
\]

During training, a diagnosis-conditioned recurrent prior regularizes the temporal evolution of the two latent trajectories. The power and FC branches retain branch-specific prior parameterizations while being coupled through a shared recurrent state.

The diagnostic label is used to condition the recurrent prior during training, but it is not provided to the variational posterior or the observation model. Therefore, ground-truth diagnostic labels are not required during test-time inference.

---

## Multi-Dynamic Generative Model

The latent logits are transformed into continuous mode-expression weights using softmax functions:

\[
\alpha_t
=
\operatorname{softmax}
\left(
\theta_t^P/\tau_{\alpha}
\right),
\qquad
\beta_t
=
\operatorname{softmax}
\left(
\theta_t^{FC}/\tau_{\beta}
\right).
\]

The time-varying regional-scale and FC matrices are constructed as

\[
G_t
=
\sum_{k=1}^{K}
\alpha_{k,t}E_k,
\qquad
F_t
=
\sum_{q=1}^{Q}
\beta_{q,t}R_q,
\]

where \(E_k\) denotes a positive diagonal regional-scale template and \(R_q\) denotes a symmetric positive-definite, unit-diagonal FC template.

The time-varying covariance matrix is

\[
C_t = G_tF_tG_t.
\]

The observation model is

\[
p_{\phi}(x_t\mid\Theta_t)
=
\mathcal{N}(x_t;0,C_t).
\]

Because \(F_t\) has unit diagonal,

\[
\operatorname{Var}(x_{t,d}\mid\Theta_t)
=
[C_t]_{dd}
=
g_{t,d}^{2},
\]

so the squared diagonal elements of \(G_t\) represent model-implied regional power, whereas \(F_t\) represents normalized interregional functional connectivity.

FC templates are parameterized using Cholesky factors followed by diagonal normalization to ensure that they remain positive definite with unit diagonal.

---

## Training Objective

CiMyGn is trained end to end using the objective

\[
\mathcal{L}
=
-\mathcal{L}_{ll}
+
\lambda_{kl}\mathcal{L}_{kl}
+
\gamma_{cls}\mathcal{L}_{cls},
\]

where

- \(\mathcal{L}_{ll}\) is the Gaussian log-likelihood term;
- \(\mathcal{L}_{kl}\) is the KL divergence between the variational posterior and the class-guided recurrent prior;
- \(\mathcal{L}_{cls}\) is the diagnostic cross-entropy loss;
- \(\lambda_{kl}\in[0,1]\) is annealed during training;
- \(\gamma_{cls}\) controls the contribution of diagnostic supervision.

The model parameters are optimized jointly using AdamW.

---

## Dataset

The experiments reported in the manuscript use five sites from the **REST-meta-MDD Consortium** that contain both MDD and healthy-control participants and share a repetition time of 2.0 s.

The retained sites are:

| Site | MDD | HC |
|---|---:|---:|
| 1 | 74 | 74 |
| 15 | 50 | 50 |
| 20 | 282 | 251 |
| 21 | 86 | 70 |
| 25 | 89 | 63 |
| **Total** | **581** | **508** |

The resulting cohort contains **1,089 participants**.

The REST-meta-MDD data are not redistributed in this repository. Users should obtain access to the dataset through the corresponding REST-meta-MDD data-access procedures.

---

## Preprocessing

The rs-fMRI data used in the manuscript were preprocessed using the DPARSF pipeline.

The preprocessing procedure includes:

- removal of initial volumes;
- slice-timing correction;
- realignment;
- normalization to MNI space;
- spatial smoothing;
- nuisance regression;
- band-pass filtering at 0.01–0.10 Hz.

Nuisance regressors include the Friston-24 motion parameters, mean white-matter and cerebrospinal-fluid signals, and linear trends.

Global signal regression was not applied.

Mean regional BOLD time series were extracted from the **90 regions of the Automated Anatomical Labeling (AAL) atlas**.

For all participants, the first

\[
T=200
\]

usable volumes were retained. Each ROI time series was z-standardized within participant across the retained sequence.

---

## Full-Rank PCA Rotation

For consistency with the Gaussian-likelihood evaluation pipeline used for the DyNeMo and MDyNeMo baselines, CiMyGn uses a fixed full-rank PCA rotation estimated from the training partition only.

This PCA operation:

- retains all \(D=90\) dimensions;
- performs no dimensionality reduction;
- performs no whitening;
- is estimated using training data only;
- is applied unchanged to the corresponding validation and test partitions;
- is used only for Gaussian likelihood evaluation;
- is not used as input to the encoder or diagnostic classifier.

Because the transformation is square and orthogonal, it preserves dimensionality and the Gaussian likelihood.

---

## Model Configuration

The retained configuration used in the manuscript is:

| Parameter | Value |
|---|---:|
| Number of ROIs \(D\) | 90 |
| Sequence length \(T\) | 200 |
| Power modes \(K\) | 3 |
| FC modes \(Q\) | 6 |
| BiLSTM layers | 2 |
| BiLSTM hidden units | 128 |
| BiLSTM inter-layer dropout | 0.2 |
| Recurrent-prior hidden units | 128 |
| Classifier hidden units | 256 |
| Classifier dropout | 0.5 |
| \(\tau_{\alpha}\) | 1 |
| \(\tau_{\beta}\) | 1 |
| \(\gamma_{cls}\) | 1 |
| Optimizer | AdamW |
| Learning rate | \(3\times10^{-4}\) |
| Weight decay | \(1\times10^{-2}\) |
| Batch size | 32 |
| Maximum epochs | 200 |
| Early-stopping patience | 30 |
| Validation fraction | 0.20 |

Early stopping and checkpoint selection are based on validation ROC-AUC.

---

## Hyperparameter Selection

Model-order selection was performed before tuning the diagnostic classification-loss weight.

With

\[
\gamma_{cls}=1,
\]

the complete grid

\[
K,Q\in\{2,\ldots,10\}
\]

was evaluated using validation performance. The retained configuration was

\[
(K,Q)=(3,6).
\]

The model-order landscape was relatively flat, and \((3,6)\) should therefore be interpreted as the best observed configuration within the prespecified search range rather than as a unique or global optimum.

After fixing

\[
(K,Q)=(3,6),
\]

the following classification-loss weights were evaluated:

\[
\gamma_{cls}
\in
\{0,0.1,0.5,1,2,5\}.
\]

The retained value was

\[
\gamma_{cls}=1.
\]

---

## Evaluation Protocols

CiMyGn is evaluated using two complementary protocols.

### 10-fold Cross-Validation

Participants are divided into ten outer folds while preserving diagnostic-class proportions.

For each outer split:

1. one fold is held out as the test partition;
2. the remaining participants form the training pool;
3. 20% of the training pool is reserved for validation.

### Leave-One-Site-Out Evaluation

For each LOSO split, all participants from one site are held out as the test partition, while participants from the remaining sites form the training pool.

Twenty percent of the training pool is reserved for validation.

For both evaluation protocols:

- the outer test partition is excluded from model selection;
- the outer test partition is excluded from early stopping;
- the outer test partition is excluded from checkpoint selection;
- all training-dependent transformations are fitted using training data only and subsequently applied unchanged to validation and test data.

The manuscript reports:

- Accuracy;
- Sensitivity;
- Specificity;
- F1-score;
- ROC-AUC.

---

## Repository Structure

```text
CiMyGn/
├── README.md
├── cimygn_model.py
├── cimygn_data.py
├── cimygn_engine.py
├── run_cimygn_cv_loso.py
├── self_test_cimygn.py
├── requirements_cimygn.txt
└── environment_cimygn_windows.yml
