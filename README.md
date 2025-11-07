# Overview
an AI model that analyzes genomic data to select significant features like variance and correlation from data and produce results on the prescence PDAC. This project implements a complete end-to-end machine learning pipeline based on the Cancer Genome Atlas Research Network findings for PDAC detection. It includes data collection, preprocessing, feature selection, model training, evaluation, and visualization.

# Key Features

Modular Architecture: Each pipeline stage is in a separate Python module for clarity and reusability
Multiple ML Models: Random Forest, Gradient Boosting, Logistic Regression, and SVM
Comprehensive Evaluation: ROC-AUC, Precision-Recall, Confusion Matrices, Feature Importance
Rich Visualizations: Publication-quality plots for all metrics
TCGA Integration: Direct API access to TCGA-PAAD genomic data

# Installation Prerequisites

Python 3.8 or higher
pip package manager

Setup
Clone or download this repository
Install required dependencies:

bashpip install -r requirements.txt

(Optional) For downloading real TCGA data, install GDC Data Transfer Tool:

bash# Instructions at: https://gdc.cancer.gov/access-data/gdc-data-transfer-tool
Usage

Running the Complete Pipeline
Run all steps in sequence:
bashpython main.py --step all
Skip data collection (use existing data):
bashpython main.py --step all --skip-collection
Running Individual Steps
Run specific pipeline steps:
bash# Data collection
python main.py --step collect

# Data preprocessing
python main.py --step preprocess

# Feature selection
python main.py --step select

# Model training
python main.py --step train

# Model evaluation
python main.py --step evaluate

# Generate visualizations
python main.py --step visualize
Running Modules Directly
Each module can also be run independently:
bashpython data_collection.py
python data_preprocessing.py
python feature_selection.py
python model_training.py
python model_evaluation.py
python visualization.py
Pipeline Stages
1. Data Collection (data_collection.py)

Queries TCGA-PAAD project from GDC Data Portal
Downloads clinical data via API
Generates manifests for gene expression and mutation data
Saves clinical annotations and metadata

Output: data/processed/clinical_data.csv
2. Data Preprocessing (data_preprocessing.py)

Loads and cleans clinical data
Integrates gene expression and mutation data
Handles missing values using imputation
Encodes categorical variables
Removes low-variance features
Detects and caps outliers
Normalizes numeric features

Output: data/processed/integrated_data.csv
3. Feature Selection (feature_selection.py)
Primary Method: Variance-Correlation Analysis
This module uses a sophisticated variance-correlation approach to identify the most significant features for PDAC detection:
Selection Process:

Variance Analysis: Calculate and rank features by variance (information content)
Target Correlation: Compute correlation with PDAC status (predictive power)
Statistical Testing: Filter by p-value significance (p < 0.05)
Redundancy Removal: Remove highly correlated features (>85%)
Combined Scoring: Weighted combination (30% variance + 70% correlation)
Final Selection: Top N features (default: 50)

Why This Approach?

Variance = Features with high variance are more informative
Correlation = Features correlated with PDAC are more predictive
Statistical Rigor = P-values ensure significance, not random associations
Non-Redundancy = Removes correlated features to avoid multicollinearity

Output:

data/processed/integrated_data_selected.csv
results/feature_importance.csv - Combined rankings
results/feature_importance_variance.csv - Variance details
results/feature_importance_correlation.csv - Correlation details

4. Model Training (model_training.py)

Splits data into train/test sets (stratified)
Trains multiple classification models:

Random Forest
Gradient Boosting
Logistic Regression
Support Vector Machine (SVM)


Performs 5-fold cross-validation
Selects best model based on ROC-AUC
Saves trained models

Output:

models/*.pkl (trained model files)
data/processed/train_data.csv
data/processed/test_data.csv

5. Model Evaluation (model_evaluation.py)

Loads trained models and test data
Generates predictions and probability scores
Calculates comprehensive metrics:

Accuracy, Precision, Recall, F1-Score
ROC-AUC
Confusion matrices
Sensitivity and Specificity


Compares all models
Extracts feature importance

Output:

results/model_performance_test.csv
results/predictions.csv
results/feature_importance_model.csv

6. Visualization (visualization.py)
Generates publication-quality visualizations:

ROC curves for all models
Precision-Recall curves
Confusion matrices (2×2 grid)
Feature importance bar chart
Model comparison chart
Class distribution plots
Prediction probability distributions

Output: results/figures/*.png
Key Biomarkers
Based on TCGA Research Network findings:
Mutation Markers

KRAS (~90% in PDAC)
TP53 (~70% in PDAC)
CDKN2A (~60% in PDAC)
SMAD4 (~55% in PDAC)
RNF43, ARID1A, TGFβR2, GNAS, RREB1, PBRM1

Expression Markers

GABRA3, IL20RB, CDK1, GPR87, TTYH3, KCNA2

DNA Repair Genes

BRCA1, BRCA2, PALB2, ATM (germline variants in 18% of cases)

Configuration
Edit config.py to customize:
Data Settings

Data directories and file paths
GDC API settings

Feature Selection (Variance-Correlation)

VARIANCE_THRESHOLD_SELECTION = 0.05 - Minimum variance to keep
VARIANCE_PERCENTILE = 75 - Keep top 75th percentile by variance
CORRELATION_WITH_TARGET_THRESHOLD = 0.15 - Minimum correlation with PDAC
FEATURE_CORRELATION_THRESHOLD = 0.85 - Remove features correlated >85%
CORRELATION_METHOD = 'pearson' - pearson or spearman
N_TOP_FEATURES = 50 - Maximum features to select

Model Training

Model hyperparameters
Cross-validation settings (5-fold)
Train/test split (80/20)

Visualization

Figure size and DPI
Color schemes

Output Files
Data Files

clinical_data.csv - Clinical annotations
integrated_data.csv - Combined genomic data
integrated_data_selected.csv - Selected features
train_data.csv - Training set
test_data.csv - Test set

Model Files

random_forest.pkl - Random Forest model
gradient_boosting.pkl - Gradient Boosting model
logistic_regression.pkl - Logistic Regression model
svm.pkl - SVM model
best_model.pkl - Best performing model

Results

model_performance.csv - Cross-validation results
model_performance_test.csv - Test set performance
predictions.csv - Model predictions
feature_importance.csv - Feature selection scores
feature_importance_model.csv - Model-based importance

Visualizations

roc_curves.png - ROC curves comparison
precision_recall_curves.png - PR curves
confusion_matrices.png - Confusion matrices grid
feature_importance.png - Top features bar chart
model_comparison.png - Metrics comparison
class_distribution.png - Train/test distribution
prediction_distributions.png - Probability distributions
variance_distribution.png - Feature variance analysis
correlation_with_target.png - Top features by correlation with PDAC
variance_vs_correlation.png - 2D scatter plot of both metrics
feature_correlation_heatmap.png - Inter-feature correlation matrix
feature_selection_summary.png - 4-panel selection summary
biomarker_importance.png - Known PDAC biomarker scores

Performance Metrics
The pipeline evaluates models using:

Accuracy: Overall classification accuracy
Precision: Positive predictive value
Recall (Sensitivity): True positive rate
F1-Score: Harmonic mean of precision and recall
ROC-AUC: Area under ROC curve
Specificity: True negative rate

Data Sources

TCGA-PAAD: The Cancer Genome Atlas - Pancreatic Adenocarcinoma
GDC Data Portal: https://portal.gdc.cancer.gov/projects/TCGA-PAAD

References

Cancer Genome Atlas Research Network. "Integrated Genomic Characterization of Pancreatic Ductal Adenocarcinoma." Cancer Cell 32.2 (2017): 185-203.
GDC Data Portal: https://gdc.cancer.gov/
TCGA PanCanAtlas: https://gdc.cancer.gov/about-data/publications/pancanatlas

Notes
The current implementation uses synthetic data for demonstration. For real analysis, download actual TCGA-PAAD data using the GDC Data Transfer Tool.
Gene expression data should be in HTSeq counts format (normalized to TPM or FPKM).
Mutation data should be in MAF (Mutation Annotation Format).
Some data in GDC requires controlled access through dbGaP.

# Troubleshooting
Issue: Module import errors
Solution: Ensure all files are in the same directory and Python path is set correctly

Issue: Missing data files
Solution: Run data collection step first or provide your own data files

Issue: Memory errors with large datasets
Solution: Adjust batch sizes in config.py or use a machine with more RAM

Issue: Plotting errors
Solution: Ensure matplotlib backend is configured correctly. Try: export MPLBACKEND=Agg

License
This project is for educational and research purposes. Please cite the original TCGA publications when using TCGA data.
Contact
