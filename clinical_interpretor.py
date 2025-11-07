import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, precision_recall_curve,
                             f1_score, accuracy_score)
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
import warnings
warnings.filterwarnings('ignore')

# For visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Matplotlib/Seaborn not available. Plotting functions disabled.")


class PDACAnomalyDetector:
    """
    PDAC Detection Model using genomic biomarkers
    """
    
    # Core biomarkers based on TCGA research
    MUTATION_GENES = [
        'KRAS', 'TP53', 'CDKN2A', 'SMAD4', 'RNF43', 'ARID1A', 
        'TGFBR2', 'GNAS', 'RREB1', 'PBRM1', 'BRAF', 'CTNNB1'
    ]
    
    EXPRESSION_GENES = [
        'GABRA3', 'IL20RB', 'CDK1', 'GPR87', 'TTYH3', 'KCNA2',
        'PTGS2', 'SP1', 'PRKCI'  # Additional prognostic markers
    ]
    
    DNA_REPAIR_GENES = [
        'BRCA1', 'BRCA2', 'PALB2', 'ATM'
    ]
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.feature_selector = None
        self.models = {}
        self.feature_importance = {}
        self.best_model = None
        
    def generate_synthetic_data(self, n_samples=500, pdac_ratio=0.3):
        """
        Generate synthetic genomic data mimicking PDAC characteristics
        
        Args:
            n_samples: Total number of samples
            pdac_ratio: Proportion of PDAC cases
        """
        np.random.seed(42)
        n_pdac = int(n_samples * pdac_ratio)
        n_normal = n_samples - n_pdac
        
        # Feature names
        mutation_features = [f'{gene}_mutation' for gene in self.MUTATION_GENES]
        expression_features = [f'{gene}_expression' for gene in self.EXPRESSION_GENES]
        dna_repair_features = [f'{gene}_mutation' for gene in self.DNA_REPAIR_GENES]
        pathway_features = ['RAS_pathway_score', 'MTOR_pathway_score', 
                           'EMT_score', 'mutation_burden']
        
        all_features = mutation_features + expression_features + dna_repair_features + pathway_features
        
        # Initialize data arrays
        data = np.zeros((n_samples, len(all_features)))
        labels = np.concatenate([np.ones(n_pdac), np.zeros(n_normal)])
        
        # PDAC samples (based on TCGA frequencies)
        # KRAS mutations: ~90% in PDAC
        data[:n_pdac, 0] = np.random.binomial(1, 0.90, n_pdac)  # KRAS
        data[:n_pdac, 1] = np.random.binomial(1, 0.70, n_pdac)  # TP53
        data[:n_pdac, 2] = np.random.binomial(1, 0.60, n_pdac)  # CDKN2A
        data[:n_pdac, 3] = np.random.binomial(1, 0.55, n_pdac)  # SMAD4
        data[:n_pdac, 4] = np.random.binomial(1, 0.25, n_pdac)  # RNF43
        data[:n_pdac, 5] = np.random.binomial(1, 0.20, n_pdac)  # ARID1A
        data[:n_pdac, 6] = np.random.binomial(1, 0.15, n_pdac)  # TGFBR2
        data[:n_pdac, 7] = np.random.binomial(1, 0.10, n_pdac)  # GNAS
        data[:n_pdac, 8:12] = np.random.binomial(1, 0.08, (n_pdac, 4))  # Other mutations
        
        # Expression levels (log2 normalized, PDAC typically upregulated)
        base_idx = len(mutation_features)
        data[:n_pdac, base_idx:base_idx+len(expression_features)] = \
            np.random.normal(2.5, 1.2, (n_pdac, len(expression_features)))
        
        # DNA repair mutations (18% germline rate from TCGA)
        dna_idx = base_idx + len(expression_features)
        data[:n_pdac, dna_idx:dna_idx+len(dna_repair_features)] = \
            np.random.binomial(1, 0.18/len(self.DNA_REPAIR_GENES), 
                              (n_pdac, len(self.DNA_REPAIR_GENES)))
        
        # Pathway scores (higher in PDAC)
        pathway_idx = dna_idx + len(dna_repair_features)
        data[:n_pdac, pathway_idx] = np.random.normal(0.75, 0.15, n_pdac)  # RAS
        data[:n_pdac, pathway_idx+1] = np.random.normal(0.65, 0.20, n_pdac)  # MTOR
        data[:n_pdac, pathway_idx+2] = np.random.normal(0.70, 0.18, n_pdac)  # EMT
        data[:n_pdac, pathway_idx+3] = np.random.normal(15, 5, n_pdac)  # Mutation burden
        
        # Normal samples (lower mutation rates, different expression)
        data[n_pdac:, 0] = np.random.binomial(1, 0.02, n_normal)  # KRAS rare in normal
        data[n_pdac:, 1:12] = np.random.binomial(1, 0.01, (n_normal, 11))
        
        # Normal expression levels (lower)
        data[n_pdac:, base_idx:base_idx+len(expression_features)] = \
            np.random.normal(0.5, 0.8, (n_normal, len(expression_features)))
        
        # Normal DNA repair
        data[n_pdac:, dna_idx:dna_idx+len(dna_repair_features)] = \
            np.random.binomial(1, 0.02, (n_normal, len(self.DNA_REPAIR_GENES)))
        
        # Normal pathway scores
        data[n_pdac:, pathway_idx] = np.random.normal(0.30, 0.10, n_normal)
        data[n_pdac:, pathway_idx+1] = np.random.normal(0.35, 0.15, n_normal)
        data[n_pdac:, pathway_idx+2] = np.random.normal(0.25, 0.12, n_normal)
        data[n_pdac:, pathway_idx+3] = np.random.normal(5, 2, n_normal)
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=all_features)
        df['PDAC_status'] = labels
        
        # Add clinical metadata
        df['age'] = np.random.normal(65, 10, n_samples).clip(40, 90)
        df['tumor_cellularity'] = np.where(
            labels == 1,
            np.random.uniform(0.10, 0.60, n_samples),  # PDAC: low cellularity
            np.random.uniform(0.70, 0.95, n_samples)   # Normal: high cellularity
        )
        
        return df
    
    def preprocess_data(self, df, target_col='PDAC_status'):
        """
        Preprocess genomic data
        """
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Identify feature types
        mutation_cols = [col for col in X.columns if '_mutation' in col]
        expression_cols = [col for col in X.columns if '_expression' in col]
        continuous_cols = [col for col in X.columns 
                          if col not in mutation_cols and '_score' in col or 'burden' in col]
        
        # Scale continuous features
        if len(expression_cols) > 0 or len(continuous_cols) > 0:
            scale_cols = expression_cols + continuous_cols
            X_scaled = X.copy()
            X_scaled[scale_cols] = self.scaler.fit_transform(X[scale_cols])
        else:
            X_scaled = X
        
        return X_scaled, y
    
    def select_features(self, X, y, k=20, method='mutual_info'):
        """
        Select top k features based on mutual information or ANOVA F-test
        """
        if method == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=min(k, X.shape[1]))
        else:
            selector = SelectKBest(f_classif, k=min(k, X.shape[1]))
        
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        self.feature_selector = selector
        
        return X_selected, selected_features
    
    def train_models(self, X_train, y_train):
        """
        Train multiple classification models
        """
        print("Training models...")
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_split=5,
                random_state=42, class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5,
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                C=1.0, penalty='l2', max_iter=1000, 
                random_state=42, class_weight='balanced'
            ),
            'SVM': SVC(
                C=1.0, kernel='rbf', probability=True,
                random_state=42, class_weight='balanced'
            )
        }
        
        # Train and evaluate each model
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        best_score = 0
        
        for name, model in models.items():
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, 
                                       cv=cv, scoring='roc_auc')
            
            # Train on full training set
            model.fit(X_train, y_train)
            self.models[name] = model
            
            print(f"\n{name}:")
            print(f"  CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            
            # Track best model
            if cv_scores.mean() > best_score:
                best_score = cv_scores.mean()
                self.best_model = name
                
            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = model.feature_importances_
        
        print(f"\nBest model: {self.best_model} (ROC-AUC: {best_score:.4f})")
    
    def evaluate_model(self, X_test, y_test, model_name=None):
        """
        Evaluate model performance
        """
        if model_name is None:
            model_name = self.best_model
        
        model = self.models[model_name]
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        print(f"\n{'='*60}")
        print(f"Evaluation Results - {model_name}")
        print(f"{'='*60}")
        
        # Metrics
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Normal', 'PDAC']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Additional metrics
        roc_auc = roc_auc_score(y_test, y_proba)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"\nROC-AUC Score: {roc_auc:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        return {
            'y_pred': y_pred,
            'y_proba': y_proba,
            'roc_auc': roc_auc,
            'accuracy': accuracy,
            'f1': f1,
            'confusion_matrix': cm
        }
    
    def predict_pdac(self, X, model_name=None):
        """
        Predict PDAC status for new samples
        """
        if model_name is None:
            model_name = self.best_model
        
        model = self.models[model_name]
        
        # Predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        
        results = pd.DataFrame({
            'PDAC_Prediction': predictions,
            'PDAC_Probability': probabilities,
            'Risk_Category': pd.cut(probabilities, 
                                   bins=[0, 0.3, 0.7, 1.0],
                                   labels=['Low', 'Moderate', 'High'])
        })
        
        return results
    
    def get_feature_importance(self, feature_names, model_name=None, top_k=15):
        """
        Get and display feature importance
        """
        if model_name is None:
            model_name = self.best_model
        
        if model_name not in self.feature_importance:
            print(f"Feature importance not available for {model_name}")
            return None
        
        importance = self.feature_importance[model_name]
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        print(f"\nTop {top_k} Most Important Features ({model_name}):")
        print("="*60)
        print(importance_df.head(top_k).to_string(index=False))
        
        return importance_df
    
    def clinical_interpretation(self, sample_data, prediction_result):
        """
        Provide clinical interpretation of predictions
        """
        prob = prediction_result['PDAC_Probability'].iloc[0]
        pred = prediction_result['PDAC_Prediction'].iloc[0]
        risk = prediction_result['Risk_Category'].iloc[0]
        
        print("\n" + "="*70)
        print("CLINICAL INTERPRETATION")
        print("="*70)
        
        print(f"\nPrediction: {'PDAC DETECTED' if pred == 1 else 'NORMAL'}")
        print(f"Confidence: {prob:.1%}")
        print(f"Risk Category: {risk}")
        
        # Key biomarker status
        print("\nKey Biomarker Status:")
        print("-" * 70)
        
        if 'KRAS_mutation' in sample_data.columns:
            kras = sample_data['KRAS_mutation'].iloc[0]
            print(f"  KRAS mutation: {'PRESENT' if kras > 0.5 else 'ABSENT'} "
                  f"(90% of PDAC cases)")
        
        if 'TP53_mutation' in sample_data.columns:
            tp53 = sample_data['TP53_mutation'].iloc[0]
            print(f"  TP53 mutation: {'PRESENT' if tp53 > 0.5 else 'ABSENT'} "
                  f"(70% of PDAC cases)")
        
        if 'CDKN2A_mutation' in sample_data.columns:
            cdkn2a = sample_data['CDKN2A_mutation'].iloc[0]
            print(f"  CDKN2A loss: {'PRESENT' if cdkn2a > 0.5 else 'ABSENT'} "
                  f"(60% of PDAC cases)")
        
        # Recommendations
        print("\nRecommendations:")
        print("-" * 70)
        if prob >= 0.7:
            print("  • HIGH RISK: Immediate further diagnostic workup recommended")
            print("  • Consider CT/MRI imaging and tumor biopsy")
            print("  • Genetic counseling if DNA repair gene mutations present")
        elif prob >= 0.3:
            print("  • MODERATE RISK: Close monitoring recommended")
            print("  • Follow-up molecular testing in 3-6 months")
            print("  • Consider additional biomarker panel")
        else:
            print("  • LOW RISK: Standard surveillance protocol")
            print("  • Annual screening for high-risk individuals")
        
        print("\n" + "="*70)


def main():
    """
    Main execution function demonstrating the PDAC detection pipeline
    """
    print("="*70)
    print("PDAC GENOMIC DETECTION MODEL")
    print("Based on Cancer Genome Atlas Research Network")
    print("="*70)
    
    # Initialize detector
    detector = PDACAnomalyDetector()
    
    # Generate synthetic data
    print("\n1. Generating synthetic genomic data...")
    df = detector.generate_synthetic_data(n_samples=500, pdac_ratio=0.3)
    print(f"   Generated {len(df)} samples ({sum(df['PDAC_status'])} PDAC cases)")
    print(f"   Features: {len(df.columns)-1}")
    
    # Preprocess data
    print("\n2. Preprocessing data...")
    X, y = detector.preprocess_data(df)
    print(f"   Shape after preprocessing: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # Feature selection
    print("\n3. Selecting important features...")
    X_train_selected, selected_features = detector.select_features(
        X_train, y_train, k=25, method='mutual_info'
    )
    X_test_selected = detector.feature_selector.transform(X_test)
    print(f"   Selected {len(selected_features)} features")
    print(f"   Top 10: {selected_features[:10]}")
    
    # Train models
    print("\n4. Training classification models...")
    detector.train_models(X_train_selected, y_train)
    
    # Evaluate best model
    print("\n5. Evaluating model performance...")
    results = detector.evaluate_model(X_test_selected, y_test)
    
    # Feature importance
    print("\n6. Analyzing feature importance...")
    importance_df = detector.get_feature_importance(selected_features, top_k=15)
    
    # Example prediction
    print("\n7. Example prediction on test sample...")
    test_sample = X_test.iloc[[0]]
    test_sample_selected = detector.feature_selector.transform(test_sample)
    prediction = detector.predict_pdac(test_sample_selected)
    detector.clinical_interpretation(test_sample, prediction)
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)
    
    return detector, results, importance_df


if __name__ == "__main__":
    detector, results, importance = main()
