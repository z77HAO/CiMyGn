# CiMyGn

## Class-Guided Generative Modeling of Brain Dynamics for Major Depressive Disorder Diagnosis from rs-fMRI

This repository provides the implementation of the Class-guided
Multi-dynamics Generative network (CiMyGn) for modeling coupled
regional power and functional connectivity (FC) dynamics in
resting-state fMRI.

CiMyGn models regional fluctuation magnitude and interregional
functional connectivity as two coupled latent dynamical processes.
A class-guided recurrent prior introduces diagnosis-conditioned
temporal regularization, while a diagnostic classifier provides
direct supervision for MDD classification.

## Repository Structure

- `cimygn_model.py`  
  Core CiMyGn model, including variational latent dynamics,
  class-guided recurrent priors, power/FC templates, covariance
  reconstruction, and diagnostic classification.

- `cimygn_data.py`  
  Data loading, quality control, within-subject standardization,
  and training-dependent preprocessing.

- `cimygn_engine.py`  
  Model training, validation, checkpoint selection, inference,
  and evaluation metrics.

- `run_cimygn_cv_loso.py`  
  Main entry point for 10-fold cross-validation and
  leave-one-site-out (LOSO) evaluation.

- `self_test_cimygn.py`  
  Lightweight implementation check using synthetic data.

- `requirements_cimygn.txt`  
  Python package requirements.

- `environment_cimygn_windows.yml`  
  Conda environment specification.

## Data

Experiments in the manuscript were conducted on five sites from
the REST-meta-MDD Consortium.

The original rs-fMRI data are not redistributed in this repository.
Users should obtain the dataset through the corresponding
REST-meta-MDD data-access procedures.

The analyses use 90 AAL ROI time series and retain the first
200 usable volumes for each participant.

## Evaluation

The manuscript evaluates CiMyGn using:

- 10-fold cross-validation;
- leave-one-site-out (LOSO) evaluation.

Training-dependent preprocessing, model selection, early stopping,
and checkpoint selection are performed without access to the
corresponding outer test partition.

## Requirements

The implementation requires Python 3.11 and PyTorch.

A Conda environment can be created using:

```bash
conda env create -f environment_cimygn_windows.yml
conda activate cimygn

