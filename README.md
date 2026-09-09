# CiMyGn

## Class-Guided Generative Modeling of Brain Dynamics for Major Depressive Disorder Diagnosis from rs-fMRI

This repository provides the implementation of the **Class-guided Multi-dynamics Generative network (CiMyGn)** for modeling diagnosis-associated brain dynamics from resting-state functional magnetic resonance imaging (rs-fMRI).

CiMyGn models regional fluctuation magnitude and functional connectivity (FC) as two distinct but coupled latent dynamical processes. A class-guided recurrent prior introduces diagnosis-conditioned temporal regularization of the coupled power and FC trajectories, while a diagnostic classifier provides direct supervision for major depressive disorder (MDD) classification.

In this framework, **regional power** refers to the model-implied conditional variance of standardized regional BOLD fluctuations. It should not be interpreted as mean BOLD intensity or frequency-domain spectral power.

---

## Method Overview

For an rs-fMRI sequence

\[
x_{1:T}=\{x_t\}_{t=1}^{T},
\]

CiMyGn uses a bidirectional long short-term memory (BiLSTM) encoder to infer separate latent trajectories for the regional-power and FC branches.

For each time point, the branch-specific latent logits are transformed into continuous mode-expression weights:

\[
\alpha_t \in \Delta^K,
\qquad
\beta_t \in \Delta^Q,
\]

where \(K\) and \(Q\) denote the numbers of regional-scale and FC templates, respectively.

The time-varying regional-scale and FC matrices are constructed as

\[
G_t=\sum_{k=1}^{K}\alpha_{k,t}E_k,
\qquad
F_t=\sum_{q=1}^{Q}\beta_{q,t}R_q,
\]

and the structured covariance is

\[
C_t = G_t F_t G_t.
\]

Here, \(G_t\) describes time-varying regional fluctuation magnitude and \(F_t\) represents normalized interregional functional connectivity.

During training, a class-guided recurrent prior regularizes the temporal evolution of the latent power and FC trajectories. The diagnostic label conditions the recurrent prior but is not provided to the variational posterior or observation model. Therefore, ground-truth diagnostic labels are not required during test-time inference.

The training objective is

\[
\mathcal{L}
=
-\mathcal{L}_{ll}
+
\lambda_{kl}\mathcal{L}_{kl}
+
\gamma_{cls}\mathcal{L}_{cls},
\]

where:

- \(\mathcal{L}_{ll}\) is the Gaussian log-likelihood term;
- \(\mathcal{L}_{kl}\) regularizes the variational posterior toward the class-guided recurrent prior;
- \(\mathcal{L}_{cls}\) is the diagnostic cross-entropy loss;
- \(\lambda_{kl}\) is annealed during training;
- \(\gamma_{cls}\) controls the contribution of diagnostic supervision.

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
