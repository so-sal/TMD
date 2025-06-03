Copy# Deep Learning Model for Temporomandibular Disorders (TMD) Diagnosis

## Overview

This repository contains the implementation of a Gated Attention Tabular Transformer (GATT) model for automated diagnosis of Temporomandibular Disorders (TMD) based on the Diagnostic Criteria for Temporomandibular Disorders (DC/TMD). Our model achieves high diagnostic accuracy across various TMD subgroups, with AUC values ranging from 0.815 to 1.000.

![Figure 1](Figure1.tif)

## Key Features

- GATT model implementation for TMD diagnosis
- Analysis of clinical signs and TMD symptoms for 28 TMD diagnostic outcomes
- Outperforms traditional machine learning models in TMD classification
- Reveals complex interrelationships between TMD signs and symptoms

## Repository Structure

This repository contains the following key components:

- **GATT.py**: Implementation of the Masked Self-Supervised Tabular Transformer (GATT). This file includes a sophisticated data generator specifically designed for tabular datasets, enabling efficient processing and augmentation of structured data.

- **utils.py**: A comprehensive utility module that encompasses:
  - Robust training and inference loops
  - Advanced statistical calculation functions
  - Feature importance analysis utilizing SHAP (SHapley Additive exPlanations) values

- **main.py**: The primary script for executing the training process. It leverages the GATT model to perform masked self-supervised learning on tabular data, showcasing the power of transformer architectures in handling structured information.

- **main_ML.py**: This script contains a variety of machine learning analyses using scikit-learn. It serves as a benchmark for comparing traditional ML approaches with our deep learning model.

- **TabNet_DeepLearning.ipynb**: A Jupyter notebook demonstrating the application of TabNet, a cutting-edge deep learning architecture specifically designed for tabular data. This notebook provides insights into the performance of TabNet on our TMD dataset.

- **AutoGluon.ipynb**: An exploratory Jupyter notebook featuring AutoGluon, an advanced AutoML framework. This notebook showcases a stacking ensemble that combines various machine learning and deep learning models, offering a comprehensive comparison of different approaches to our TMD classification task.

These files collectively represent a state-of-the-art approach to TMD diagnosis, combining traditional machine learning techniques with advanced deep learning methodologies.


## Model Performance


### DC/TMD subgroup (Right)

| Diagnoses | AUROC (95% CI) | Accuracy (95% CI) | Sensitivity (95% CI) | Specificity (95% CI) | PPV (95% CI) | NPV (95% CI) | Positive cases (n=929) |
|-----------|----------------|-------------------|----------------------|----------------------|--------------|--------------|------------------------|
| Myalgia | 0.830 (0.801-0.858) | 0.763 (0.734-0.790) | 0.763 (0.712-0.811) | 0.763 (0.731-0.797) | 0.574 (0.525-0.628) | 0.885 (0.858-0.911) | 189 |
| Local Myalgia | 0.817 (0.787-0.847) | 0.745 (0.716-0.773) | 0.780 (0.723-0.831) | 0.733 (0.699-0.767) | 0.499 (0.449-0.550) | 0.907 (0.882-0.928) | 63 |
| Myofascial Pain | 0.934 (0.907-0.961) | 0.868 (0.847-0.888) | 0.927 (0.869-0.973) | 0.861 (0.838-0.883) | 0.434 (0.366-0.500) | 0.990 (0.982-0.996) | 68 |
| Myofascial Pain with Referral | 0.997 (0.995-1.000) | 0.975 (0.964-0.986) | 0.984 (0.944-1.000) | 0.975 (0.964-0.985) | 0.732 (0.631-0.831) | 0.999 (0.996-1.000) | 87 |
| DD with Reduction | 1.000 (1.000-1.000) | 0.998 (0.995-1.000) | 1.000 (1.000-1.000) | 0.997 (0.992-1.000) | 0.993 (0.982-1.000) | 1.000 (1.000-1.000) | 103 |
| DD with Intermittent Locking | 0.945 (0.931-0.958) | 0.860 (0.835-0.882) | 0.993 (0.978-1.000) | 0.834 (0.806-0.859) | 0.539 (0.478-0.591) | 0.998 (0.995-1.000) | 363 |
| DD without Reduction with Limited Opening | 0.991 (0.982-0.999) | 0.958 (0.944-0.970) | 0.952 (0.889-1.000) | 0.958 (0.944-0.971) | 0.625 (0.533-0.723) | 0.996 (0.992-1.000) | 388 |
| DD without Reduction without Limited Opening | 0.978 (0.956-1.000) | 0.948 (0.933-0.962) | 0.977 (0.941-1.000) | 0.945 (0.929-0.961) | 0.649 (0.565-0.731) | 0.997 (0.994-1.000) | 138 |
| Arthralgia | 1.000 (1.000-1.000) | 0.997 (0.992-1.000) | 0.992 (0.982-1.000) | 1.000 (1.000-1.000) | 1.000 (1.000-1.000) | 0.995 (0.988-1.000) | 261 |
| DJD | 0.998 (0.994-1.000) | 0.996 (0.990-0.999) | 0.986 (0.963-1.000) | 0.997 (0.994-1.000) | 0.986 (0.961-1.000) | 0.997 (0.994-1.000) | 77 |
| HATMD | 1.000 (1.000-1.000) | 0.999 (0.997-1.000) | 1.000 (1.000-1.000) | 0.999 (0.996-1.000) | 0.987 (0.955-1.000) | 1.000 (1.000-1.000) | 82 |

### DC/TMD subgroup (Left)

| Diagnoses | AUROC (95% CI) | Accuracy (95% CI) | Sensitivity (95% CI) | Specificity (95% CI) | PPV (95% CI) | NPV (95% CI) | Positive cases (n=929) |
|-----------|----------------|-------------------|----------------------|----------------------|--------------|--------------|------------------------|
| Myalgia | 0.821 (0.792-0.849) | 0.721 (0.692-0.751) | 0.786 (0.738-0.838) | 0.693 (0.655-0.730) | 0.526 (0.478-0.575) | 0.882 (0.852-0.911) | 274 |
| Local Myalgia | 0.815 (0.784-0.846) | 0.797 (0.771-0.821) | 0.623 (0.559-0.683) | 0.857 (0.831-0.883) | 0.601 (0.536-0.668) | 0.868 (0.843-0.891) | 281 |
| Myofascial Pain | 0.972 (0.962-0.982) | 0.893 (0.873-0.913) | 0.988 (0.958-1.000) | 0.884 (0.862-0.906) | 0.449 (0.377-0.522) | 0.999 (0.995-1.000) | 236 |
| Myofascial Pain with Referral | 0.999 (0.998-1.000) | 0.986 (0.977-0.992) | 0.991 (0.969-1.000) | 0.985 (0.976-0.993) | 0.900 (0.843-0.949) | 0.999 (0.996-1.000) | 239 |
| DD with Reduction | 0.999 (0.996-1.000) | 0.998 (0.995-1.000) | 1.000 (1.000-1.000) | 0.996 (0.990-1.000) | 0.995 (0.987-1.000) | 1.000 (1.000-1.000) | 96 |
| DD with Intermittent Locking | 0.925 (0.908-0.941) | 0.837 (0.813-0.859) | 0.963 (0.932-0.987) | 0.805 (0.776-0.831) | 0.558 (0.500-0.609) | 0.988 (0.978-0.996) | 81 |
| DD without Reduction with Limited Opening | 1.000 (1.000-1.000) | 0.999 (0.997-1.000) | 1.000 (1.000-1.000) | 0.999 (0.996-1.000) | 0.986 (0.954-1.000) | 1.000 (1.000-1.000) | 61 |
| DD without Reduction without Limited Opening | 0.999 (0.998-1.000) | 0.984 (0.975-0.991) | 0.990 (0.969-1.000) | 0.983 (0.974-0.992) | 0.879 (0.820-0.936) | 0.999 (0.996-1.000) | 109 |
| Arthralgia | 1.000 (1.000-1.000) | 0.995 (0.990-0.999) | 0.995 (0.987-1.000) | 0.994 (0.988-1.000) | 0.992 (0.983-1.000) | 0.996 (0.991-1.000) | 288 |
| DJD | 1.000 (1.000-1.000) | 0.998 (0.995-1.000) | 1.000 (1.000-1.000) | 0.997 (0.992-1.000) | 0.992 (0.981-1.000) | 1.000 (1.000-1.000) | 385 |
| HATMD | 1.000 (1.000-1.000) | 0.998 (0.995-1.000) | 1.000 (1.000-1.000) | 0.998 (0.994-1.000) | 0.976 (0.939-1.000) | 1.000 (1.000-1.000) | 152 |

### Other TMD Diagnoses

| Diagnoses | AUROC (95% CI) | Accuracy (95% CI) | Sensitivity (95% CI) | Specificity (95% CI) | PPV (95% CI) | NPV (95% CI) | Positive cases (n=929) |
|-----------|----------------|-------------------|----------------------|----------------------|--------------|--------------|------------------------|
| Subluxation | 0.991 (0.985-0.996) | 0.931 (0.915-0.947) | 0.986 (0.953-1.000) | 0.927 (0.909-0.944) | 0.519 (0.429-0.605) | 0.999 (0.996-1.000) | 69 |
| Arthrogenous TMD | 0.825 (0.798-0.852) | 0.763 (0.735-0.791) | 0.607 (0.562-0.653) | 0.924 (0.899-0.946) | 0.891 (0.854-0.924) | 0.696 (0.660-0.733) | 471 |
| Myogenous TMD | 0.996 (0.990-1.000) | 0.995 (0.989-0.999) | 0.995 (0.989-1.000) | 0.992 (0.978-1.000) | 0.997 (0.991-1.000) | 0.989 (0.974-1.000) | 664 |
| Mixed TMD | 0.904 (0.885-0.922) | 0.806 (0.783-0.834) | 0.825 (0.786-0.864) | 0.794 (0.758-0.827) | 0.727 (0.688-0.772) | 0.872 (0.843-0.901) | 372 |
| HATMD | 1.000 (1.000-1.000) | 1.000 (1.000-1.000) | 1.000 (1.000-1.000) | 1.000 (1.000-1.000) | 1.000 (1.000-1.000) | 1.000 (1.000-1.000) | 126 |
| ADD | 0.998 (0.996-1.000) | 0.989 (0.983-0.996) | 0.992 (0.983-1.000) | 0.986 (0.975-0.996) | 0.988 (0.977-0.996) | 0.991 (0.981-1.000) | 489 |
