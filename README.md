# Motor Insurance Claim Frequency Prediction — Warta Data Science Competition

Report from a student data science competition organized by Warta TUiR. The task was to predict and rank motor third-party liability insurance policies by expected claim frequency, using anonymized historical insurance data provided by the organizer.

**Team:** Bartosz Eppel, Mateusz Gonera, Jakub Marczak.

**Result:** First place in the competition, achieving an organizer-reported **25.22% normalized Gini** on the hidden test set.

[Read the project report](report.pdf)

## Approach

- Engineered features from historical insurance policies and claims data, working with a main dataset of **491,229 policy records**.
- Developed an ensemble of **15 Poisson LightGBM models** using **565 features**, with Optuna hyperparameter tuning and five-fold cross-validation across three random seeds.
- Applied **out-of-fold target encoding** with exposure-weighted smoothing to handle categorical variables while reducing target leakage.
- Combined model predictions through percentile-rank averaging and evaluated performance using normalized Gini.
- Addressed target leakage by excluding target-period exposure from predictors and incorporating it as an offset in Poisson modeling.

## My Main Contribution

- Developed **client- and vehicle-level features from historical insurance policies**, capturing claim-free history, coverage continuity, policy duration, and customer behavior. **Seven of the ten most important features** in the final model originated from policy history.
- Implemented **out-of-fold smoothed target encoding**, using exposure-weighted claim-frequency estimates to handle categorical variables while limiting target leakage and overfitting.
- Experimented with **different feature sets and modeling approaches**, including classification and Poisson objectives, and performed **LightGBM hyperparameter optimization with Optuna**.
- Evaluated model performance using **normalized Gini** and analyzed training and scoring feature distributions using **Kolmogorov–Smirnov statistics**.

The competition result was achieved collaboratively by the team. The project represents a modeling proof of concept rather than a deployed insurance pricing system. Normalized Gini measures ranking performance and does not directly assess probability calibration.

## Repository Contents

This repository contains the complete competition report and project summary. The original datasets and model implementation are not publicly available, as the competition was conducted in the organizer's closed environment.

The report documents the modeling methodology, feature engineering, validation strategy, and experimental results. Performance on future policy periods was not established during the competition.
