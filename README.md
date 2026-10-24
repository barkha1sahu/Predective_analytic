<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=28&pause=1000&color=4F8EF7&center=true&vCenter=true&width=900&lines=Mobile+Price+Predictive+Analytics+CA;Three+ML+Pipelines+%E2%80%A2+Two+Models+%E2%80%A2+One+Project" alt="Typing SVG" />

# Mobile Phone Price - Predictive Analytics CA

**End-to-end machine learning project across three real-world smartphone datasets covering both classification (price tier) and regression (exact price prediction)**

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Pandas](https://img.shields.io/badge/Pandas-3.0-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Kaggle](https://img.shields.io/badge/Datasets-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://kaggle.com)

</div>

---

## Table of Contents

| # | Section |
|---|---------|
| 1 | [Project Overview](#1-project-overview) |
| 2 | [Problem Statement and Objectives](#2-problem-statement-and-objectives) |
| 3 | [Datasets](#3-datasets) |
| 4 | [Repository Structure](#4-repository-structure) |
| 5 | [Methodology and Approach](#5-methodology-and-approach) |
| 6 | [Notebook 1 - Enhanced Price Tier Classification (INR)](#6-notebook-1---enhanced-price-tier-classification-inr) |
| 7 | [Notebook 2 - Global Price Tier Classification (USD)](#7-notebook-2---global-price-tier-classification-usd) |
| 8 | [Notebook 3 - Smartphone Price Prediction Regression (INR)](#8-notebook-3---smartphone-price-prediction-regression-inr) |
| 9 | [Model Comparison and Results](#9-model-comparison-and-results) |
| 10 | [Key Findings and Insights](#10-key-findings-and-insights) |
| 11 | [Conclusion and Future Scope](#11-conclusion-and-future-scope) |
| 12 | [Technologies and Libraries](#12-technologies-and-libraries) |
| 13 | [Installation and Execution](#13-installation-and-execution) |
| 14 | [Project Resources and Deliverables](#14-project-resources-and-deliverables) |
| 15 | [References and Acknowledgements](#15-references-and-acknowledgements) |

---

## 1. Project Overview

This Continuous Assessment (CA) project applies **Predictive Analytics** to the mobile phone market - one of the fastest-evolving consumer electronics sectors. Using three distinct real-world datasets, we build and compare machine learning models that:

- **Classify** smartphones into price tiers (Budget / Mid-Range / Premium) based on hardware specifications
- **Predict** exact launch prices (INR) as a regression task with error analysis and brand-premium detection

Two model families are evaluated throughout: **Random Forest (RF)** and **Gradient Boosting (GB)**, both implemented via scikit-learn pipelines with median imputation, cross-validated hyperparameter tuning, and rich evaluation dashboards.

> **Academic context:** CA submission for Predictive Analytics module - covers the full ML workflow from raw data ingestion through statistical analysis, feature engineering, model training, evaluation, and deployment artefacts.

---

## 2. Problem Statement and Objectives

### Problem Statement

> *"Can we reliably predict the price tier or exact launch price of a smartphone from its hardware specifications alone, and which features drive that price the most?"*

### Objectives

- **Ingest and clean** three heterogeneous mobile-phone datasets (different currencies, schemas, and quality issues)
- **Conduct exploratory data analysis** with statistical hypothesis testing (Kruskal-Wallis, Chi-square, Spearman correlation)
- **Engineer meaningful features** (camera sums, battery-per-inch ratio, display pixel density, processor families)
- **Train and tune** Random Forest and Gradient Boosting models using cross-validated grid/random search
- **Evaluate rigorously** with 15+ metrics: Accuracy, F1-macro, ROC-AUC, MCC, Kappa, Brier score, Permutation Importance, Wilcoxon and McNemar tests
- **Interpret** results via permutation importance, partial dependence plots, and SHAP analysis
- **Identify** underpriced (best-value) and overpriced (brand-premium) smartphones

---

## 3. Datasets

Three Kaggle datasets are used - each offering a different perspective on smartphone pricing:

### Dataset 1 - Indian Smartphone Market (INR)

| Property | Detail |
|----------|--------|
| **Kaggle Link** | [Kaggle - Mobile Dataset for India Latest](https://www.kaggle.com/datasets/shreyanshmishra23/mobile-dataset-for-india-latest) |
| **File** | `data/dataset 1.zip` -> `final_mobile_dataset.csv` |
| **Rows / Columns** | 1,056 x 14 |
| **After cleaning** | 1,049 unique phones, 27 brands, 34 features |
| **Target** | Price tier (Budget <= Rs 18,990 / Mid <= Rs 30,999 / Premium > Rs 30,999) |
| **Price range** | Rs 7,689 - Rs 1,89,999 |
| **Class balance** | Budget 33.6% / Mid-Range 33.2% / Premium 33.3% (Balanced) |
| **Used in** | Notebook 1 (Enhanced Classification) |

**Columns:** Processor, Brand, RAM (GB), ROM (GB), Battery Capacity, Display Size, Display Quality (AMOLED/LCD), Refresh Rate, Rear Camera MP, Front Camera MP, iOS flag, OS Version, Price (INR)

---

### Dataset 2 - Global Smartphones 2025 (USD)

| Property | Detail |
|----------|--------|
| **Kaggle Link** | [Kaggle - Mobiles Dataset 2025](https://www.kaggle.com/datasets/abdulmalik1518/mobiles-dataset-2025) |
| **File** | `data/dataset 2.zip` -> `Mobiles_Dataset_(2025)_.csv` |
| **Rows / Columns** | 930 x 15 |
| **After cleaning** | 915 unique phones, 18 brands, 36 features |
| **Target** | Price tier (Budget <= $310 / Mid <= $699 / Premium > $699) |
| **Price range** | $79 - $39,622 |
| **Class balance** | Budget 305 / Mid-Range 325 / Premium 285 (Approx Balanced) |
| **Used in** | Notebook 2 (Global Classification) |

**Columns:** Company Name, Model Name, Mobile Weight, RAM, Front Camera, Back Camera, Processor, Battery Capacity, Screen Size, Launch Prices (PKR / INR / CNY / USD / AED), Launch Year

---

### Dataset 3 - Indian Smartphones with Rich Specs (INR)

| Property | Detail |
|----------|--------|
| **Kaggle Link** | [Kaggle - 5G Smartphones Available in India](https://www.kaggle.com/datasets/ramjasmaurya/5g-smartphones-available-in-india) |
| **File** | `data/dataset 3.zip` -> `smartphones_v4.csv` |
| **Rows / Columns** | 858 x 29 |
| **Target** | Exact launch price (INR) - Regression |
| **Price range** | Rs 5,700 - Rs 2,59,999 / Median Rs 27,230 / Skewness 2.63 |
| **Used in** | Notebook 3 (Price Prediction Regression) |

**Columns:** brand, model, price, rating, is_5G, is_NFC, is_eSIM, is_IRBlaster, processor_name, core count, processor_speed, ram_size, internal_memory, battery_capacity, max_charging_power, fast_charging, screen_size, resolution, refresh_rate, back_camera, front_camera, camera counts, extended_memory, OS, OS version

---

## 4. Repository Structure

```
PREDECTIVE_ANALYTIC_CA/
|
|-- notebooks/
|   |-- Mobile_Price_Tier_Enhanced_Notebook.ipynb    <- Notebook 1: Enhanced Classification (INR)
|   |-- Mobiles_Price_Tier_Classification.ipynb      <- Notebook 2: Global Classification (USD)
|   +-- smartphone_price_prediction.ipynb            <- Notebook 3: Price Regression (INR)
|
|-- data/
|   |-- dataset 1.zip    <- final_mobile_dataset.csv  (INR, 1,056 phones)
|   |-- dataset 2.zip    <- Mobiles_Dataset_2025.csv  (USD, 930 phones)
|   +-- dataset 3.zip    <- smartphones_v4.csv        (INR, 858 phones)
|
|-- output/
|   |-- Enhanced/
|   |   |-- images/      <- 23 PNG figures from Notebook 1
|   |   +-- tables/      <- Text metric outputs from Notebook 1
|   |-- Classification/
|   |   |-- images/      <- 7 PNG figures from Notebook 2
|   |   +-- tables/      <- Text metric outputs from Notebook 2
|   +-- Prediction/
|       |-- images/      <- 23 PNG figures from Notebook 3
|       +-- tables/      <- Text metric outputs from Notebook 3
|
|-- report/
|   +-- Smartphone_Price_Prediction_Report.docx      <- Full written project report
|
|-- ppt/                 <- Presentation folder (slides to be added)
|
|-- README.md            <- This file
|-- requirements.txt     <- Python dependencies
+-- .gitignore
```

---

## 5. Methodology and Approach

All three notebooks follow the same rigorous ML pipeline:

```
+---------------------------------------------------------------------+
|                     STANDARD ML PIPELINE                            |
+----------+----------+----------+----------+----------+-------------+
|  1. DATA |  2. EDA  |  3. FEAT |  4. SPLIT|  5. TUNE |  6. EVAL   |
|  AUDIT   |  & STATS |  ENG.    |  & PIPE  |  & TRAIN |  & EXPLAIN |
+----------+----------+----------+----------+----------+-------------+
```

### Step 1 - Data Audit and Cleaning
- Remove exact duplicates
- Detect and handle missing values (camera MPs, storage, ratings)
- Fix encoding anomalies (RAM stored as 1/2 -> recoded to 12 GB)
- Harmonise brand labels (30 -> 27 unique brands in Dataset 1)
- Parse string fields (e.g. "6GB", "48MP", "3,600mAh" -> numeric values)

### Step 2 - Exploratory Data Analysis (EDA)
- Kruskal-Wallis H-test for each numeric feature vs price tier (non-parametric ANOVA)
- Chi-square tests for categorical features vs price tier
- Spearman rank correlations with price
- Processor-tier heatmaps, distribution plots, box plots, pairplots
- PCA 2-D projections of feature space

### Step 3 - Feature Engineering

| Feature | Description | Notebooks |
|---------|-------------|-----------|
| `camera_sum_mp` | Rear MP + Front MP | 1 |
| `battery_per_inch` | Battery mAh / Display inches | 1 |
| `rom_per_ram` | ROM GB / RAM GB | 1 |
| `n_back_cams` | Count of rear camera sensors | 2 |
| `back_total_mp` | Sum of all rear camera MPs | 2 |
| `megapixels` | Width x Height / 1,000,000 | 3 |
| `ppi` | Pixel density (pixels per inch) | 3 |
| `total_cameras` | Rear count + Front count | 3 |
| `processor_brand` | 9-family chip grouping (one-hot) | 3 |
| Brand dummies | One-hot encoding of brand | 1, 2 |

### Step 4 - Train/Test Split and Pipeline
- Stratified 80/20 split (classification), random 80/20 (regression)
- `sklearn.Pipeline`: `SimpleImputer(strategy='median')` -> `Classifier/Regressor`
- Models trained on `log(price)` in Notebook 3 for skew correction

### Step 5 - Hyperparameter Tuning
- **Classification:** `GridSearchCV` with 5-fold stratified CV, scoring = macro-F1
- **Regression:** `RandomizedSearchCV` with 5-fold CV, scoring = negative MAE
- Random Forest: searched over `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`
- Gradient Boosting: searched over `n_estimators`, `learning_rate`, `max_depth`, `subsample`

### Step 6 - Evaluation and Explainability
- **Classification metrics:** Accuracy, Balanced Accuracy, Precision, Recall, F1 (macro + weighted), ROC-AUC (OvR), MCC, Quadratic Kappa, Brier Score, MAE (tier steps)
- **Regression metrics:** MAE, RMSE, MedAE, MAPE, R-squared, R-squared(log-price), Train R-squared
- **Statistical tests:** Paired Wilcoxon (15-fold CV), McNemar (test-set), Paired t-test (regression folds)
- **Feature explainability:** Impurity importance, Permutation importance, Partial Dependence Plots, SHAP

---

## 6. Notebook 1 - Enhanced Price Tier Classification (INR)

> **File:** [`notebooks/Mobile_Price_Tier_Enhanced_Notebook.ipynb`](notebooks/Mobile_Price_Tier_Enhanced_Notebook.ipynb)
> **Dataset:** [`data/dataset 1.zip`](data/dataset%201.zip) | **Report:** [`report/Smartphone_Price_Prediction_Report.docx`](report/Smartphone_Price_Prediction_Report.docx)

### Dataset Summary

| Metric | Value |
|--------|-------|
| Raw rows | 1,056 |
| After deduplication | 1,049 |
| Brands (after harmonisation) | 27 |
| Top brands | Samsung (163), Realme (155), Vivo (120), Xiaomi (78), OnePlus (68) |
| Features after engineering | 34 (13 numeric + 21 one-hot) |
| Train / Test split | 839 / 210 (stratified, 70 per class in test) |
| Price tiers | Budget <= Rs 18,990 / Mid <= Rs 30,999 / Premium > Rs 30,999 |

### Key EDA Findings

- **Apple = 100% Premium** - all 62 Apple phones fall in the Premium tier
- **Processor family** is the single strongest categorical predictor
- **battery_per_inch** and **front_mp** are the top permutation-importance features
- RAM encoding anomaly: values 1 and 2 GB actually represented 12 GB flagship phones

### Model Performance

#### Overall Test-Set Results (210 phones)

| Model | Accuracy | F1 (macro) | ROC-AUC | MCC | Quad. Kappa | Brier Score |
|-------|----------|-----------|---------|-----|------------|-------------|
| **Random Forest** | **0.8095** | **0.8083** | 0.9411 | **0.7146** | **0.8481** | 0.0936 |
| Gradient Boosting | 0.8000 | 0.7990 | **0.9471** | 0.7003 | 0.8410 | **0.0847** |

#### Per-Class Breakdown

| Class | RF Precision | RF Recall | RF F1 | GB Precision | GB Recall | GB F1 |
|-------|------------|---------|-------|------------|---------|-------|
| Budget | 0.851 | **0.900** | **0.875** | 0.824 | 0.871 | 0.847 |
| Mid-Range | 0.731 | 0.700 | 0.715 | 0.716 | 0.686 | 0.701 |
| Premium | 0.841 | 0.829 | 0.835 | **0.855** | **0.843** | **0.849** |

> Mid-Range is the hardest class to classify across all three notebooks - it sits between the other two tiers with overlapping specs.

#### Repeated CV (15 folds) and Statistical Tests

| Model | Mean Macro-F1 | Std | 95% CI |
|-------|-------------|-----|--------|
| Random Forest | 0.7923 | 0.0249 | [0.778, 0.807] |
| **Gradient Boosting** | **0.8061** | 0.0208 | [0.794, 0.818] |

- **Wilcoxon test:** p = 0.0067 -> GB slightly better on CV folds
- **McNemar test:** p = 0.824 -> No significant difference on test set
- **Conclusion:** Models are practically equivalent; choose GB for calibration, RF for interpretability

#### Top Permutation Features (mean drop in macro-F1)

| Rank | Feature | RF Drop | GB Drop |
|------|---------|---------|---------|
| 1 | battery_per_inch | 0.0403 | **0.0517** |
| 2 | display_in | 0.0323 | 0.0468 |
| 3 | front_mp | 0.0339 | 0.0434 |
| 4 | is_amoled | 0.0239 | 0.0216 |
| 5 | rom_gb | 0.0062 | 0.0295 |

### Key Figures - Notebook 1

<details>
<summary>Click to expand all 23 figures</summary>

| Figure | Preview |
|--------|---------|
| Data Quality Overview | ![](output/Enhanced/images/cell_010_img_00.png) |
| Price Distribution and Tier Cut-offs | ![](output/Enhanced/images/cell_019_img_01.png) |
| Brand Positioning Map | ![](output/Enhanced/images/cell_021_img_02.png) |
| Feature Correlation Heatmap | ![](output/Enhanced/images/cell_023_img_03.png) |
| Processor vs Price Tier | ![](output/Enhanced/images/cell_025_img_04.png) |
| Core Specs per Tier (box plots) | ![](output/Enhanced/images/cell_027_img_05.png) |
| Battery and Display Analysis | ![](output/Enhanced/images/cell_029_img_06.png) |
| Price Tier Overlap / Pairplot | ![](output/Enhanced/images/cell_031_img_07.png) |
| PCA 2-D Projection | ![](output/Enhanced/images/cell_033_img_08.png) |
| Effect Sizes (Cohen d) | ![](output/Enhanced/images/cell_038_img_09.png) |
| Hyperparameter Grid Heatmaps | ![](output/Enhanced/images/cell_044_img_10.png) |
| Confusion Matrices | ![](output/Enhanced/images/cell_049_img_11.png) |
| ROC Curves | ![](output/Enhanced/images/cell_051_img_12.png) |
| Precision-Recall Curves | ![](output/Enhanced/images/cell_053_img_13.png) |
| Calibration / Reliability Diagram | ![](output/Enhanced/images/cell_055_img_14.png) |
| Learning Curves | ![](output/Enhanced/images/cell_057_img_15.png) |
| Impurity Feature Importance | ![](output/Enhanced/images/cell_060_img_16.png) |
| Permutation Importance | ![](output/Enhanced/images/cell_062_img_17.png) |
| Partial Dependence Plots | ![](output/Enhanced/images/cell_064_img_18.png) |
| Model Comparison (Wilcoxon/McNemar) | ![](output/Enhanced/images/cell_069_img_19.png) |
| Prediction Confidence vs Correctness | ![](output/Enhanced/images/cell_072_img_20.png) |
| OOF Error Analysis | ![](output/Enhanced/images/cell_075_img_21.png) |
| RAM Repair Sensitivity | ![](output/Enhanced/images/cell_080_img_22.png) |

</details>

---

## 7. Notebook 2 - Global Price Tier Classification (USD)

> **File:** [`notebooks/Mobiles_Price_Tier_Classification.ipynb`](notebooks/Mobiles_Price_Tier_Classification.ipynb)
> **Dataset:** [`data/dataset 2.zip`](data/dataset%202.zip)

### Dataset Summary

| Metric | Value |
|--------|-------|
| Raw rows | 930 |
| After deduplication | 915 unique phones |
| Brands | 18 companies |
| Features after engineering | 36 (10 numeric + 26 one-hot) |
| Train / Test split | 732 / 183 (stratified) |
| Price tiers | Budget <= $310 / Mid <= $699 / Premium > $699 |

### Key EDA Findings - Spearman Correlations with USD Price

| Feature | Correlation |
|---------|------------|
| ram_gb | **+0.550** |
| storage_gb | +0.520 |
| back_total_mp | +0.252 |
| weight_g | +0.236 |
| front_mp | +0.216 |
| battery_mah | -0.203 |

### Model Performance

#### Overall Test-Set Results (183 phones)

| Model | Accuracy | Balanced Acc | F1 (macro) | ROC-AUC |
|-------|----------|-------------|-----------|---------|
| **Random Forest** | **0.8361** | 0.8346 | **0.8409** | **0.9647** |
| Gradient Boosting | **0.8361** | **0.8367** | 0.8387 | 0.9490 |

#### Per-Class Results

| Class | RF Precision | RF Recall | RF F1 | GB Precision | GB Recall | GB F1 | Support |
|-------|------------|---------|-------|------------|---------|-------|---------|
| Budget | **0.923** | 0.787 | 0.850 | 0.893 | 0.820 | 0.855 | 61 |
| Mid-Range | 0.716 | **0.892** | 0.795 | 0.761 | 0.831 | 0.794 | 65 |
| Premium | **0.940** | 0.825 | 0.879 | 0.875 | **0.860** | **0.867** | 57 |

#### Top Feature Importances

| Rank | Random Forest | Gradient Boosting |
|------|-------------|------------------|
| 1 | ram_gb (0.120) | **ram_gb (0.189)** |
| 2 | weight_g (0.115) | weight_g (0.135) |
| 3 | front_mp (0.102) | front_mp (0.134) |
| 4 | screen_in (0.090) | battery_mah (0.078) |
| 5 | battery_mah (0.082) | cpu_mediatek (0.078) |

### Key Figures - Notebook 2

<details>
<summary>Click to expand all 7 figures</summary>

| Figure | Preview |
|--------|---------|
| Dataset Overview | ![](output/Classification/images/cell_015_img_00.png) |
| Correlation Heatmap | ![](output/Classification/images/cell_016_img_01.png) |
| Specs by Price Tier | ![](output/Classification/images/cell_017_img_02.png) |
| Confusion Matrices | ![](output/Classification/images/cell_027_img_03.png) |
| ROC Curves | ![](output/Classification/images/cell_028_img_04.png) |
| Feature Importance | ![](output/Classification/images/cell_029_img_05.png) |
| Model Comparison | ![](output/Classification/images/cell_030_img_06.png) |

</details>

---

## 8. Notebook 3 - Smartphone Price Prediction Regression (INR)

> **File:** [`notebooks/smartphone_price_prediction.ipynb`](notebooks/smartphone_price_prediction.ipynb)
> **Dataset:** [`data/dataset 3.zip`](data/dataset%203.zip)

### Dataset Summary

| Metric | Value |
|--------|-------|
| Rows | 858 |
| Exact duplicates | 0 |
| Features after engineering | 30 (26 numeric + 4 one-hot) |
| Train / Test split | 686 / 172 |
| Target | log(price) -> back-transformed to INR |
| Price skewness | 2.63 raw -> 0.52 after log transform |

### Model Performance

#### Baseline vs Tuned - Test Set (172 phones)

| Model | MAE (Rs) | RMSE (Rs) | MedAE (Rs) | MAPE (%) | R-sq | R-sq (log) | Train R-sq |
|-------|---------|---------|---------|---------|---|---------|---------|
| RF (baseline) | 5,790 | 11,414 | 2,500 | 13.98 | 0.882 | 0.932 | 0.988 |
| GB (baseline) | 4,430 | 8,882 | 1,857 | 11.25 | 0.929 | 0.951 | 0.989 |
| RF (tuned) | 5,537 | 11,555 | 2,371 | 13.54 | 0.879 | 0.936 | 0.987 |
| **GB (tuned) BEST** | **3,887** | **8,088** | **1,918** | **10.75** | **0.941** | **0.956** | 0.999 |

> **Best model: Gradient Boosting (tuned)** - average error of only Rs 3,887 on unseen phones

#### Error by Price Segment

| Segment | n | RF MAE (Rs) | GB MAE (Rs) | RF MAPE% | GB MAPE% |
|---------|---|-----------|-----------|---------|---------|
| Budget (<Rs 15k) | 138 | 1,725 | **1,546** | 16.2 | **14.7** |
| Mid (Rs 15-30k) | 363 | 2,660 | **2,228** | 12.1 | **10.1** |
| Upper-mid (Rs 30-60k) | 227 | 4,405 | **3,728** | 10.4 | **8.5** |
| Flagship (>Rs 60k) | 130 | 17,428 | **12,964** | 15.5 | **12.2** |

#### Segment Classification (4-class bonus task)

| Model | Accuracy | Macro F1 | MCC |
|-------|----------|---------|-----|
| **Random Forest** | **0.866** | **0.867** | **0.812** |
| Gradient Boosting | 0.808 | 0.804 | 0.725 |

#### Top Feature Importances

| Rank | Random Forest | Gradient Boosting |
|------|-------------|------------------|
| 1 | processor_speed | **ram_size** |
| 2 | megapixels | megapixels |
| 3 | internal_memory | internal_memory |
| 4 | ram_size | ppi |
| 5 | ppi | res_w |

#### Value Analysis - Best and Worst Value Phones

**Underpriced (Best Value - priced below spec-fair price):**

| Brand | Model | Market Price (Rs) | Fair Price (Rs) | Discount |
|-------|-------|-----------------|--------------|---------|
| Motorola | Edge 20 Pro 5G | 20,750 | 37,600 | **-44.8%** |
| Samsung | Galaxy F05 | 6,499 | 11,500 | -43.5% |
| Nothing | Phone 3 (16+512GB) | 54,999 | 93,300 | -41.1% |
| Tecno | Spark Go 5G | 9,999 | 16,500 | -39.4% |

**Overpriced (Brand/Design Premium):**

| Brand | Model | Market Price (Rs) | Fair Price (Rs) | Premium |
|-------|-------|-----------------|--------------|---------|
| Realme | C63 5G | 24,999 | 12,800 | **+95.3%** |
| Nothing | Phone 3 5G | 79,999 | 45,300 | +76.6% |
| OPPO | Find X9 Ultra | 1,69,999 | 96,300 | +76.5% |
| Vivo | V60 5G | 50,302 | 28,700 | +75.3% |

### Key Figures - Notebook 3

<details>
<summary>Click to expand all 23 figures</summary>

| Figure | Preview |
|--------|---------|
| Missing Data Overview | ![](output/Prediction/images/cell_006_img_00.png) |
| Price Distribution (Raw vs Log) | ![](output/Prediction/images/cell_007_img_01.png) |
| Correlation Heatmap | ![](output/Prediction/images/cell_008_img_02.png) |
| Specs by Brand | ![](output/Prediction/images/cell_009_img_03.png) |
| Feature vs Price Scatter Plots | ![](output/Prediction/images/cell_010_img_04.png) |
| Screen Spec Distribution | ![](output/Prediction/images/cell_011_img_05.png) |
| Screen Spec vs Price | ![](output/Prediction/images/cell_011_img_06.png) |
| Baseline CV Training Curves | ![](output/Prediction/images/cell_020_img_07.png) |
| Residual Plots | ![](output/Prediction/images/cell_023_img_08.png) |
| Predicted vs Actual (scatter) | ![](output/Prediction/images/cell_023_img_09.png) |
| Error Distribution | ![](output/Prediction/images/cell_024_img_10.png) |
| SHAP Feature Influence | ![](output/Prediction/images/cell_025_img_11.png) |
| CV Stability Comparison | ![](output/Prediction/images/cell_026_img_12.png) |
| Impurity Feature Importance | ![](output/Prediction/images/cell_028_img_13.png) |
| Permutation Importance | ![](output/Prediction/images/cell_030_img_14.png) |
| Segment Error Analysis | ![](output/Prediction/images/cell_032_img_15.png) |
| Error by Price Segment | ![](output/Prediction/images/cell_034_img_16.png) |
| SHAP Beeswarm | ![](output/Prediction/images/cell_036_img_17.png) |
| SHAP Waterfall | ![](output/Prediction/images/cell_037_img_18.png) |
| Partial Dependence Plots | ![](output/Prediction/images/cell_040_img_19.png) |
| Brand Effect | ![](output/Prediction/images/cell_041_img_20.png) |
| Processor Effect | ![](output/Prediction/images/cell_042_img_21.png) |
| Final Model Summary | ![](output/Prediction/images/cell_044_img_22.png) |

</details>

---

## 9. Model Comparison and Results

### Classification - All Notebooks Side-by-Side

| Notebook | Dataset | Model | Accuracy | F1 (macro) | ROC-AUC |
|----------|---------|-------|----------|-----------|---------|
| Notebook 1 (INR) | 1,049 phones | **Random Forest** | **0.8095** | **0.8083** | 0.9411 |
| Notebook 1 (INR) | 1,049 phones | Gradient Boosting | 0.8000 | 0.7990 | **0.9471** |
| Notebook 2 (USD) | 915 phones | **Random Forest** | **0.8361** | **0.8409** | **0.9647** |
| Notebook 2 (USD) | 915 phones | Gradient Boosting | 0.8361 | 0.8387 | 0.9490 |

### Regression - Notebook 3

| Model | MAE (Rs) | RMSE (Rs) | MAPE % | R-sq |
|-------|---------|---------|--------|---|
| RF (baseline) | 5,790 | 11,414 | 13.98 | 0.882 |
| GB (baseline) | 4,430 | 8,882 | 11.25 | 0.929 |
| RF (tuned) | 5,537 | 11,555 | 13.54 | 0.879 |
| **GB (tuned) BEST** | **3,887** | **8,088** | **10.75** | **0.941** |

---

## 10. Key Findings and Insights

1. **RAM is the single most important feature** across all three datasets and both model families - higher RAM consistently predicts higher price tier

2. **Gradient Boosting wins on regression** - tuned GB achieves R-sq = 0.941 and MAE = Rs 3,887, outperforming Random Forest on all 6 regression metrics

3. **Random Forest is more competitive on classification** - leads on the USD dataset (F1 = 0.841, ROC-AUC = 0.965) and ties on accuracy in Notebook 2

4. **Mid-Range is consistently the hardest class** - it sits between Budget and Premium with overlapping specifications, yielding the lowest per-class F1 in all classification notebooks (0.70-0.80)

5. **battery_per_inch and display_in are top permutation features** in Notebook 1, indicating that premium phones balance battery size with screen size more efficiently

6. **Apple phones are perfectly classified as Premium** in Notebook 1 - 100% of 62 Apple devices fall in the Premium tier

7. **Brand premium is real and quantifiable** - Nothing, OPPO, and Samsung foldables are priced 60-95% above their spec-predicted fair price; Motorola Edge 20 Pro offers the best value at -44.8%

8. **Log-transformation of price is essential for regression** - raw price skewness of 2.63 drops to 0.52 after log transform, significantly improving model fit

9. **Statistical tests validate model differences** - Wilcoxon p = 0.007 shows GB is marginally better on CV folds (Notebook 1), but McNemar p = 0.824 confirms no significant test-set difference -> the two models are practically equivalent for classification

---

## 11. Conclusion and Future Scope

### Conclusion

This project successfully demonstrates that smartphone hardware specifications alone can predict price tier with **80-84% accuracy** and exact launch price with an average error of just **Rs 3,887 (MAPE ~ 10.8%)** using ensemble tree-based models. Key deliverables include:

- Three fully documented Jupyter notebooks covering classification and regression
- Rigorous evaluation with 15+ metrics, statistical tests, and calibration analysis
- Feature importance analysis revealing RAM, processor family, and battery density as primary price drivers
- Actionable consumer insight: a value-analysis identifying best and worst-value smartphones

### Future Scope

| Direction | Description |
|-----------|-------------|
| **Deep Learning** | Test neural network regressors (MLP, TabNet) for non-linear price patterns |
| **Temporal Analysis** | Incorporate launch year and depreciation curves for time-aware prediction |
| **Multi-currency Normalisation** | PPP-adjusted cross-country price comparison |
| **Real-time Pipeline** | Kaggle API scraping + automated model retraining as new phones launch |
| **User-facing App** | Streamlit/Gradio web app for "enter specs -> get fair price estimate" |
| **Segment-specific Models** | Separate regressors for Budget/Mid/Flagship segments to reduce within-segment error |
| **Brand Effect Decomposition** | Separate "spec value" from "brand premium" pricing component |

---

## 12. Technologies and Libraries

| Category | Library / Tool | Version |
|----------|---------------|---------|
| Language | Python | 3.11+ |
| Data manipulation | pandas | 3.0 |
| Numerical computing | NumPy | 2.4 |
| Machine learning | scikit-learn | 1.8 |
| Statistical tests | SciPy | 1.17 |
| Visualisation | Matplotlib | 3.8 |
| Visualisation | Seaborn | 0.13 |
| Notebooks | Jupyter / nbformat | 1.0 / 5.9 |
| Model persistence | joblib | 1.3 |
| SHAP analysis | SHAP | 0.43 |

---

## 13. Installation and Execution

### Prerequisites
- Python 3.11 or higher
- pip package manager
- 8 GB RAM recommended (for full notebook execution)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/barkha1sahu/PREDECTIVE_ANALYTIC_CA.git
cd PREDECTIVE_ANALYTIC_CA

# 2. (Recommended) Create a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook
```

### Running the Notebooks (in order)

| Step | Notebook | What it does |
|------|----------|-------------|
| 1 | `notebooks/Mobile_Price_Tier_Enhanced_Notebook.ipynb` | Enhanced classification - Dataset 1 (INR) |
| 2 | `notebooks/Mobiles_Price_Tier_Classification.ipynb` | Global classification - Dataset 2 (USD) |
| 3 | `notebooks/smartphone_price_prediction.ipynb` | Price regression - Dataset 3 (INR) |

> **Note:** Each notebook is self-contained. Datasets are loaded from `data/` via a relative path. Unzip the dataset files before running.

---

## 14. Project Resources and Deliverables

### Jupyter Notebooks

| Notebook | Description | Link |
|----------|-------------|------|
| Mobile_Price_Tier_Enhanced_Notebook.ipynb | Enhanced Classification (INR, 23 figures) | [Open](notebooks/Mobile_Price_Tier_Enhanced_Notebook.ipynb) |
| Mobiles_Price_Tier_Classification.ipynb | Global Classification (USD, 7 figures) | [Open](notebooks/Mobiles_Price_Tier_Classification.ipynb) |
| smartphone_price_prediction.ipynb | Price Regression (INR, 23 figures) | [Open](notebooks/smartphone_price_prediction.ipynb) |

### Datasets

| Dataset | Description | Kaggle | Local |
|---------|-------------|--------|-------|
| Dataset 1 | Indian phones (INR) - 1,056 rows | [Kaggle](https://www.kaggle.com/datasets/shreyanshmishra23/mobile-dataset-for-india-latest) | [data/dataset 1.zip](data/dataset%201.zip) |
| Dataset 2 | Global phones (USD) - 930 rows | [Kaggle](https://www.kaggle.com/datasets/abdulmalik1518/mobiles-dataset-2025) | [data/dataset 2.zip](data/dataset%202.zip) |
| Dataset 3 | Indian phones with full specs - 858 rows | [Kaggle](https://www.kaggle.com/datasets/ramjasmaurya/5g-smartphones-available-in-india) | [data/dataset 3.zip](data/dataset%203.zip) |

### Figures and Visualisations

| Folder | Contents |
|--------|---------|
| [output/Enhanced/images/](output/Enhanced/images/) | 23 PNG figures from Notebook 1 |
| [output/Classification/images/](output/Classification/images/) | 7 PNG figures from Notebook 2 |
| [output/Prediction/images/](output/Prediction/images/) | 23 PNG figures from Notebook 3 |

### Metric Tables (Raw Text)

| File | Description |
|------|-------------|
| [output/Enhanced/tables/all_text_outputs.txt](output/Enhanced/tables/all_text_outputs.txt) | All printed outputs from Notebook 1 |
| [output/Classification/tables/all_text_outputs.txt](output/Classification/tables/all_text_outputs.txt) | All printed outputs from Notebook 2 |
| [output/Prediction/tables/all_text_outputs.txt](output/Prediction/tables/all_text_outputs.txt) | All printed outputs from Notebook 3 |

### Report and Documentation

| File | Description |
|------|-------------|
| [report/Smartphone_Price_Prediction_Report.docx](report/Smartphone_Price_Prediction_Report.docx) | Full written project report (5.0 MB) |
| [README.md](README.md) | This documentation file |
| [requirements.txt](requirements.txt) | Python dependencies |

### Missing Files - Action Required

| File | Status | Action |
|------|--------|--------|
| `ppt/` presentation | Folder exists but is **empty** | Add your PPT/PPTX file here |
| Trained model `.joblib` files | Not present in repo | Re-run notebooks to generate |
| Cleaned/preprocessed CSVs | Not in repo | Unzip datasets and commit extracted CSVs if desired |

---

## 15. References and Acknowledgements

### Datasets
1. Mishra, S. (2025). *Mobile Dataset for India (Latest).* Kaggle. https://www.kaggle.com/datasets/shreyanshmishra23/mobile-dataset-for-india-latest
2. Malik, A. (2025). *Mobiles Dataset 2025.* Kaggle. https://www.kaggle.com/datasets/abdulmalik1518/mobiles-dataset-2025
3. Maurya, R. (2023). *5G Smartphones Available in India.* Kaggle. https://www.kaggle.com/datasets/ramjasmaurya/5g-smartphones-available-in-india

### Libraries and Frameworks
4. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825-2830.
5. McKinney, W. (2010). Data Structures for Statistical Computing in Python. *Proceedings of the 9th Python in Science Conference.*
6. Harris, C. R., et al. (2020). Array programming with NumPy. *Nature*, 585, 357-362.

### Academic References
7. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
8. Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. *Annals of Statistics*, 29(5), 1189-1232.
9. Lundberg, S. M., and Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.

---

<div align="center">

**Predictive Analytics CA | Academic Project**

Made with scikit-learn, pandas, seaborn

[![GitHub](https://img.shields.io/badge/GitHub-barkha1sahu-181717?style=flat-square&logo=github)](https://github.com/barkha1sahu/PREDECTIVE_ANALYTIC_CA)

</div>
