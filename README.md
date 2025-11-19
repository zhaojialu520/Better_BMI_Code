# Better BMI: Novel Digital Anthropometry Equations for Adiposity Assessment in Chinese Population

A comprehensive machine learning pipeline for predicting DXA-derived adiposity relative indices from 3dO body scan measurements.

## 📋 Project Overview

This repository contains a complete analytical workflow for developing and validating prediction models that estimate DXA-derived adiposity relative indices using digital anthropometric phenotypes.

## 🏗️ Project Structure
├── Data preprocessing.R # Data cleaning and preparation

├── Model Training

│ ├── Machine learning model comparison and hyperparameter optimization.py # Multiple ML algorithms comparison and Performance metrics of the machine learning algorithms on
the training set.

│ ├── Wilcoxon-RankSum Test.R # Statistical significance testing

│ ├── FSLR model training.py # Forward Stepwise Linear Regression

│ ├── VIF Selected Feature Procedure.R # Feature selection with VIF


├── Model assessment

│ ├── Comparison with BMI.py # Benchmark against BMI

│ ├── DW Test.R # Durbin-Watson test for autocorrelation

│ ├── Normality Test.R # Distribution normality checks

│ ├── SHAP analysis.py # Model interpretability analysis

│ ├── Subgroup analysis.py # Stratified performance analysis




├── Figure Generation

│ └── Regression plot & Bland-Altman.py # Visualization and validation plots


└── README.md

## 🎯 Target Variables

The models predict the following DXA-derived body composition parameters:
- **VATmass**: Visceral Adipose Tissue mass
- **FM**: Fat Mass
- **FMI**: Fat Mass Index
- **LM**: Lean Mass
- **Android**: Android region fat
- **Gynoid**: Gynoid region fat
- **A_G**: Android-to-Gynoid ratio
- **BFP**: Body Fat Percentage

## 🔧 Installation & Requirements

### Prerequisites
- Python 3.11.4
- R 4.3.2

### Python Packages
numpy (1.24.3), pandas (1.5.3), scipy (1.10.1), scikit-learn (1.3.0), statsmodels (0.14.0), matplotlib (3.7.1), seaborn (0.12.2), and shap (0.48.0).

### R Packages
tidyverse (2.0.0), dplyr (1.1.4), tidyr (1.3.1), broom (1.0.9), car (3.1.3), lmtest (0.9.40), sandwich (3.1.1), moments (0.14.1), nortest (1.0.4), tseries(0.10.58), ggplot2 (3.5.2), patchwork (1.3.2), and caret (7.0.1).




