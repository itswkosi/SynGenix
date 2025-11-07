import os

# Project directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# GDC API Configuration
GDC_API_BASE = "https://api.gdc.cancer.gov"
PROJECT_ID = "TCGA-PAAD"

# Data types to download (open access only)
DATA_TYPES = {
    'clinical': 'clinical',
    'gene_expression': 'Gene Expression Quantification',
    'mutations': 'Masked Somatic Mutation',
    'copy_number': 'Copy Number Segment'
}

# TCGA-PAAD Key Biomarkers (from literature)
KEY_MUTATION_GENES = [
    'KRAS', 'TP53', 'CDKN2A', 'SMAD4', 'RNF43', 'ARID1A',
    'TGFBR2', 'GNAS', 'RREB1', 'PBRM1', 'BRAF', 'CTNNB1'
]

KEY_EXPRESSION_GENES = [
    'GABRA3', 'IL20RB', 'CDK1', 'GPR87', 'TTYH3', 'KCNA2',
    'PTGS2', 'SP1', 'PRKCI'
]

DNA_REPAIR_GENES = ['BRCA1', 'BRCA2', 'PALB2', 'ATM']

# Data preprocessing parameters
MISSING_VALUE_THRESHOLD = 0.5  # Remove features with >50% missing values
VARIANCE_THRESHOLD = 0.01  # Remove low variance features
OUTLIER_ZSCORE_THRESHOLD = 3.5

# Feature selection parameters
N_TOP_FEATURES = 50
FEATURE_SELECTION_METHOD = 'variance_correlation'  # Primary method

# Variance-based selection
VARIANCE_THRESHOLD_SELECTION = 0.05  # Higher threshold for feature selection
VARIANCE_PERCENTILE = 75  # Select top 75th percentile by variance

# Correlation-based selection
CORRELATION_WITH_TARGET_THRESHOLD = 0.15  # Minimum correlation with PDAC status
FEATURE_CORRELATION_THRESHOLD = 0.85  # Remove features correlated >0.85
CORRELATION_METHOD = 'pearson'  # Options: 'pearson', 'spearman'

# Model training parameters
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Model hyperparameters
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'class_weight': 'balanced'
    },
    'gradient_boosting': {
        'n_estimators': 150,
        'learning_rate': 0.05,
        'max_depth': 5,
        'subsample': 0.8
    },
    'logistic_regression': {
        'C': 1.0,
        'penalty': 'l2',
        'max_iter': 2000,
        'class_weight': 'balanced'
    },
    'svm': {
        'C': 1.0,
        'kernel': 'rbf',
        'gamma': 'scale',
        'class_weight': 'balanced'
    }
}

# Visualization parameters
FIGURE_DPI = 300
FIGURE_SIZE = (10, 8)

# File paths
CLINICAL_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'clinical_data.csv')
GENE_EXPRESSION_FILE = os.path.join(PROCESSED_DATA_DIR, 'gene_expression.csv')
MUTATION_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'mutation_data.csv')
INTEGRATED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'integrated_data.csv')
TRAIN_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'train_data.csv')
TEST_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'test_data.csv')

# Results file paths
FEATURE_IMPORTANCE_FILE = os.path.join(RESULTS_DIR, 'feature_importance.csv')
MODEL_PERFORMANCE_FILE = os.path.join(RESULTS_DIR, 'model_performance.csv')
PREDICTIONS_FILE = os.path.join(RESULTS_DIR, 'predictions.csv')

print(f"Configuration loaded. Data directory: {DATA_DIR}")
