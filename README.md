# Stacking Regression for Antenna Design
## EC-101 Academic Project — National Institute of Technology (NIT) Hamirpur

**Author:** Manav Hans  
**Roll Number:** 26BCS063  
**Course:** EC-101 [B.Tech Semester 1]  
**Department:** Electronics and Communication Engineering  
**Allotment Category:** Multi-Classical Regression (Item #63)  
**Assigned Algorithm:** Stacking Regression with Meta-Model Study  
**Primary Deliverable:** Documented Python Code + Experimental Results + Academic Report + Streamlit Dashboard  

---

## 📌 Project Overview

In high-frequency wireless communications, designing microstrip patch and slot antennas requires extensive numerical electromagnetic simulations using full-wave solvers (such as CST Microwave Studio, HFSS, or FEKO). These numerical simulations solve Maxwell’s partial differential equations over millions of meshed volumetric elements, requiring hours to days for multi-parameter sweeps.

This project implements a machine learning **Forward Surrogate Model** using **Stacking Regression** to predict **35 antenna performance outputs** across 5 discrete resonance frequencies ($F_1$ to $F_5$) using **10 physical and geometric antenna design variables** ($DV_1$ to $DV_{10}$).

### Problem Formulation:
$$\mathbf{X} \in \mathbb{R}^{10} \xrightarrow{\quad\mathcal{F}_{\text{Stacking}}\quad} \mathbf{Y} \in \mathbb{R}^{35}$$

- **Input Features (10 Design Variables):** $DV_1, DV_2, \dots, DV_{10}$
- **Target Outputs (35 Performance Indicators):**
  - $P_1\_F_1$ to $P_7\_F_1$ (Frequency 1)
  - $P_1\_F_2$ to $P_7\_F_2$ (Frequency 2)
  - $P_1\_F_3$ to $P_7\_F_3$ (Frequency 3)
  - $P_1\_F_4$ to $P_7\_F_4$ (Frequency 4)
  - $P_1\_F_5$ to $P_7\_F_5$ (Frequency 5)

---

## 🏗️ Stacking Architecture & Methodology

Unlike simple ensemble averaging, voting, or blending, this project implements a genuine **Stacking Regression Architecture (Wolpert, 1992)** with rigorous **Out-of-Fold (OOF)** prediction generation:

1. **Level-0 Base Models (Diverse Ensembles):**
   - **Random Forest Regressor** (Non-linear Bagging)
   - **Extra Trees Regressor** (Extremely Randomized Trees)
   - **Gradient Boosting Regressor** (Sequential Residual Boosting)
   - **K-Nearest Neighbors Regressor** (Non-parametric Distance-weighted)
   - **Support Vector Regression (SVR)** (RBF Kernel Maximum Margin)
   - **Ridge Regression** (Linear L2 Regularized Baseline)

2. **Out-of-Fold (OOF) K-Fold Cross-Validation:**
   - The 70% training partition ($N=35$) is split into $K=5$ folds.
   - For each fold, base models are trained on 4 folds ($K-1$) and predict the held-out fold.
   - This ensures **zero data leakage** into the meta-model, as every training sample receives predictions from models that never saw that sample during training.

3. **Level-1 Meta-Learner & Meta-Model Study:**
   - Compares **Ridge Regression ($L_2$ regularized)** vs. **Linear Regression (OLS)** vs. **ElasticNet**.
   - Solves multicollinearity among base regressors through shrinkage.
   - Selected based strictly on **validation set performance**.

4. **Multi-Output Support:**
   - Handles the 35 target variables via multi-output regression pipelines.

---

## 📁 Repository Directory Structure

```text
antenna-stacking-regression/
├── data/
│   └── dataset.csv                 # Official EC-101 Antenna Design Dataset (50 designs x 51 cols)
├── models/
│   └── saved_model/
│       ├── final_stacking_model.joblib  # Trained Multi-Output Stacking Regressor
│       └── preprocessor.joblib          # Fitted StandardScaler and metadata
├── results/
│   ├── metrics/
│   │   ├── test_summary_metrics.csv     # Model comparison table on Test Set
│   │   ├── test_per_target_metrics.csv  # Detailed metrics for all 35 outputs
│   │   ├── test_raw_metrics.json        # Raw JSON evaluation metrics
│   │   ├── val_summary_metrics.csv      # Validation summary metrics
│   │   └── meta_model_study_comparison.csv # Ridge vs OLS vs ElasticNet comparison
│   └── plots/
│       ├── actual_vs_predicted.png      # Scatter plots with y=x ideal line
│       ├── residual_distribution.png    # Residual histogram and KDE curve
│       ├── mae_comparison.png           # MAE comparison bar chart
│       ├── rmse_comparison.png          # RMSE comparison bar chart
│       ├── r2_comparison.png            # R2 score comparison bar chart
│       ├── individual_r2_scores.png     # Individual R2 for all 35 outputs
│       ├── correlation_heatmap.png      # Feature & Target correlation heatmap
│       ├── base_vs_stacking.png         # Multi-metric grouped comparison
│       ├── meta_model_comparison.png    # Meta-Model study bar chart
│       └── feature_importance.png       # Gini importance of DV1-DV10
├── src/
│   ├── preprocessing.py            # Data loading, cleaning, validation, scaling
│   ├── stacking.py                 # MultiOutputStackingRegressor class & OOF CV
│   ├── train.py                    # Master training, meta-model study & evaluation
│   ├── evaluate.py                 # Evaluation metrics calculation (MAE, RMSE, R2)
│   ├── predict.py                  # CLI inference engine for new DV1-DV10 inputs
│   └── visualizations.py           # Publication-quality figure generation (10 plots)
├── app.py                          # Interactive Streamlit Web Application
├── main.py                         # Master orchestration script
├── requirements.txt                # Python package dependencies
├── README.md                       # Comprehensive setup and user guide
└── report.md                       # Complete 25-section Academic Project Report
```

---

## ⚙️ Installation & Setup Instructions

### Prerequisites
- Python 3.9, 3.10, or 3.11 installed.
- VS Code (Visual Studio Code) or any terminal environment.
- Git (optional, for cloning).

### 1. Running in VS Code on Windows

Open PowerShell or Command Prompt in the project root directory:

```powershell
# 1. Navigate to the project root
cd antenna-stacking-regression

# 2. Create a virtual environment (recommended)
python -m venv venv

# 3. Activate the virtual environment
# In PowerShell:
.\venv\Scripts\Activate.ps1
# In Command Prompt (cmd):
.\venv\Scripts\activate.bat

# 4. Upgrade pip and install all required dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Running on macOS / Linux

```bash
cd antenna-stacking-regression
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Execution Guide

### Option 1: Run the Complete Master Pipeline
Executes preprocessing, 5-fold cross-validation OOF generation, meta-model comparison, model selection, evaluation, visualization generation, and sample inference:

```bash
python main.py --mode all
```

### Option 2: Preprocessing and Training Only
```bash
python src/train.py
```

### Option 3: Run Inference on New Antenna Variables
Predict all 35 performance outputs from 10 design variables:

```bash
# Run prediction on built-in sample antenna design:
python src/predict.py --sample

# Run batch predictions from a CSV containing DV1 to DV10:
python src/predict.py --csv path/to/input_dvs.csv --output predictions.csv
```

### Option 4: Launch the Streamlit Web Application
Launch the interactive graphical engineering dashboard:

```bash
streamlit run app.py
```
After running, your default web browser will open `http://localhost:8501`.

---

## 📊 Summary of Actual Experimental Results

All results were calculated from actual execution on the untouched Test Set (15% split, 8 samples):

| Model | MAE | RMSE | $R^2$ | Training Time (s) | Prediction Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Stacking Regressor (Ridge $\alpha=20.0$)** | **0.9443** | **1.6059** | **0.0923** | 2.7535 | 0.01437 |
| **Extra Trees Regressor** | 0.9510 | 1.5959 | 0.1554 | 0.0242 | 0.00249 |
| **Gradient Boosting Regressor** | 0.9874 | 1.7849 | -0.0114 | 0.3765 | 0.00521 |
| **K-Nearest Neighbors (KNN)** | 1.0239 | 1.7747 | 0.0219 | 0.0002 | 0.00030 |
| **Support Vector Regressor (SVR)** | 1.0317 | 1.7960 | -0.0449 | 0.0092 | 0.00273 |
| **Random Forest Regressor** | 1.0544 | 1.7604 | 0.1359 | 0.0393 | 0.00255 |
| **Ridge Regression (Base)** | 1.2784 | 2.6727 | 0.1902 | 0.0004 | 0.00006 |

### Meta-Model Study Comparison:
- **Ridge ($\alpha=20.0$):** Validation MAE = **1.2060** | Test MAE = **0.9443** *(Selected Best)*
- **Ridge ($\alpha=10.0$):** Validation MAE = 1.2329 | Test MAE = 0.9653
- **ElasticNet ($\alpha=0.1$):** Validation MAE = 1.2759 | Test MAE = 1.0021
- **Ridge ($\alpha=1.0$):** Validation MAE = 1.2937 | Test MAE = 1.0333
- **Linear Regression (OLS):** Validation MAE = 1.3186 | Test MAE = 1.0708

---

## 📦 Submission Deliverables

1. Source Code (`src/`, `main.py`, `app.py`)
2. Saved Model Binaries (`models/saved_model/*.joblib`)
3. Evaluation CSVs and JSON (`results/metrics/`)
4. 10 High-Resolution Plots (`results/plots/*.png`)
5. Streamlit Application (`app.py`)
6. Comprehensive Academic Report (`report.md`)
7. Documentation (`README.md`, `requirements.txt`)
