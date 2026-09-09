# CiMyGn

## Class-Guided Generative Modeling of Brain Dynamics for Major Depressive Disorder Diagnosis from rs-fMRI

This repository provides the implementation of the **Class-guided Multi-dynamics Generative network (CiMyGn)** for modeling diagnosis-associated brain dynamics from resting-state functional magnetic resonance imaging (rs-fMRI).

CiMyGn models regional power and functional connectivity (FC) as two distinct but coupled latent dynamical processes. A class-guided recurrent prior regularizes their temporal evolution, while an end-to-end classifier provides direct diagnostic supervision for major depressive disorder (MDD).

Here, **regional power** denotes the model-implied conditional variance of standardized regional BOLD fluctuations rather than mean BOLD intensity or frequency-domain spectral power.

## Method

A BiLSTM encoder infers separate latent-logit trajectories for the power and FC branches. During training, a diagnosis-conditioned recurrent prior regularizes their temporal evolution. The latent logits are transformed into continuous mode-expression weights and used to construct

\[
G_t=\sum_{k=1}^{K}\alpha_{k,t}E_k,\qquad
F_t=\sum_{q=1}^{Q}\beta_{q,t}R_q,
\]

with the time-varying covariance

\[
C_t=G_tF_tG_t.
\]

The model is trained using

\[
\mathcal{L}
=
-\mathcal{L}_{ll}
+\lambda_{kl}\mathcal{L}_{kl}
+\gamma_{cls}\mathcal{L}_{cls},
\]

where the KL weight is annealed during training.

## Data

Experiments use five sites (1, 15, 20, 21, and 25) from the **REST-meta-MDD Consortium**, comprising **581 participants with MDD and 508 healthy controls (1,089 participants total)**.

Mean BOLD time series are extracted from the **90 AAL regions**, and the first **200 usable volumes** are retained for each participant. Each ROI time series is z-standardized within participant.

The REST-meta-MDD data are not redistributed in this repository.

## Experimental Configuration

| Parameter | Value |
|---|---:|
| ROIs | 90 |
| Sequence length | 200 |
| Power modes \(K\) | 3 |
| FC modes \(Q\) | 6 |
| BiLSTM layers | 2 |
| BiLSTM hidden units | 128 |
| BiLSTM dropout | 0.2 |
| Recurrent-prior hidden units | 128 |
| Classifier hidden units | 256 |
| Classifier dropout | 0.5 |
| \(\tau_{\alpha},\tau_{\beta}\) | 1 |
| \(\gamma_{cls}\) | 1 |
| Optimizer | AdamW |
| Learning rate | \(3\times10^{-4}\) |
| Weight decay | \(1\times10^{-2}\) |
| Batch size | 32 |
| Maximum epochs | 200 |
| Early-stopping patience | 30 |
| Validation fraction | 0.20 |

A fixed full-rank PCA rotation is estimated from the training partition only and is used exclusively for Gaussian-likelihood evaluation; it performs neither dimensionality reduction nor whitening.

Model-order selection evaluates \(K,Q\in\{2,\ldots,10\}\) with \(\gamma_{cls}=1\), retaining \((K,Q)=(3,6)\). After fixing the model order, \(\gamma_{cls}\in\{0,0.1,0.5,1,2,5\}\) is evaluated, with \(\gamma_{cls}=1\) retained.

## Evaluation

CiMyGn is evaluated using **10-fold cross-validation** and **leave-one-site-out (LOSO)** evaluation. In each outer split, 20% of the training pool is reserved for validation. The outer test set is not used for preprocessing estimation, model selection, early stopping, or checkpoint selection.

Reported classification metrics include Accuracy, Sensitivity, Specificity, F1-score, and ROC-AUC.

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
