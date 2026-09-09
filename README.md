# CiMyGn

## Class-Guided Generative Modeling of Brain Dynamics for Major Depressive Disorder Diagnosis from rs-fMRI

This repository contains the core implementation of the **Class-guided Multi-dynamics Generative network (CiMyGn)** for modeling diagnosis-associated brain dynamics from resting-state functional magnetic resonance imaging (rs-fMRI).

CiMyGn models regional power and functional connectivity (FC) as distinct but coupled latent dynamical processes. A class-guided recurrent prior regularizes their temporal evolution, while a diagnostic classifier provides direct supervision for major depressive disorder (MDD).

Here, **regional power** denotes the model-implied conditional variance of standardized regional BOLD fluctuations, rather than mean BOLD intensity or frequency-domain spectral power.

## Method

A two-layer BiLSTM encoder infers separate latent trajectories for the power and FC branches. The class-guided recurrent prior regularizes their joint temporal evolution. The corresponding mode-expression weights construct

$$
G_t = \sum_{k=1}^{K} \alpha_{k,t} E_k,
\qquad
F_t = \sum_{q=1}^{Q} \beta_{q,t} R_q,
$$

with the time-varying covariance

$$
C_t = G_t F_t G_t.
$$

The training objective is

$$
\mathcal{L} =
-\mathcal{L}_{ll}
+\lambda_{kl}\mathcal{L}_{kl}
+\gamma_{cls}\mathcal{L}_{cls},
$$

where the KL weight is annealed during training.

## Data

Experiments were conducted on five REST-meta-MDD sites (1, 15, 20, 21, and 25), including **581 participants with MDD and 508 healthy controls (1,089 participants in total)**.

Mean BOLD time series were extracted from **90 AAL regions**. The first **200 usable volumes** were retained, and each ROI time series was z-standardized within participant.

The REST-meta-MDD data are not redistributed in this repository.

## Evaluation

CiMyGn is evaluated using **10-fold cross-validation** and **leave-one-site-out (LOSO)** evaluation. In each outer split, 20% of the training pool is reserved for validation. The outer test partition is excluded from preprocessing estimation, model selection, early stopping, and checkpoint selection.

Reported metrics include Accuracy, Sensitivity, Specificity, F1-score, and ROC-AUC.

## Installation

Using Conda:

```bash
conda env create -f environment_cimygn_windows.yml
conda activate cimygn
```

Alternatively, install the required packages using pip:

```bash
pip install -r requirements_cimygn.txt
```

## Citation

If you use this code, please cite:

Q. Zhao, X. Zhang, W. Yuan, Z. Wu, X. Zhang, and B. Hu,  
“Class-Guided Generative Modeling of Brain Dynamics for Major Depressive Disorder Diagnosis from rs-fMRI.”
