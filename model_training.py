import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score
import joblib
import config
import os

class ModelTrainer:
    """
    Trains multiple ML models for PDAC detection
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.cv_results = {}
    
    def load_data(self, filepath=None):
        """
        Load selected features dataset
        """
        if filepath is None:
            filepath = config.INTEGRATED_DATA_FILE.replace('.csv', '_selected.csv')
        
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Loaded {df.shape[0]} samples with {df.shape[1]} features")
        
        return df
    
    def prepare_train_test_split(self, df, target_col='is_tumor'):
        """
        Split data into train and test sets
        """
        print("\nSplitting data into train/test sets...")
        
        # Separate features and target
        X = df.drop(columns=['case_id', target_col])
        y = df[target_col]
        
        # Split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Training class distribution: {y_train.value_counts().to_dict()}")
        print(f"Test class distribution: {y_test.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def initialize_models(self):
        """
        Initialize ML models with configured parameters
        """
        print("\nInitializing models...")
        
        self.models = {
            'Random Forest': RandomForestClassifier(
                **config.MODEL_PARAMS['random_forest'],
                random_state=config.RANDOM_STATE,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                **config.MODEL_PARAMS['gradient_boosting'],
                random_state=config.RANDOM_STATE
            ),
            'Logistic Regression': LogisticRegression(
                **config.MODEL_PARAMS['logistic_regression'],
                random_state=config.RANDOM_STATE
            ),
            'SVM': SVC(
                **config.MODEL_PARAMS['svm'],
                probability=True,
                random_state=config.RANDOM_STATE
            )
        }
        
        print(f"Initialized {len(self.models)} models")
        
        return self.models
    
    def cross_validate_models(self, X_train, y_train):
        """
        Perform cross-validation for all models
        """
        print("\n" + "="*70)
        print("CROSS-VALIDATION")
        print("="*70)
        
        cv = StratifiedKFold(
            n_splits=config.CV_FOLDS,
            shuffle=True,
            random_state=config.RANDOM_STATE
        )
        
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'f1': make_scorer(f1_score),
            'roc_auc': make_scorer(roc_auc_score)
        }
        
        best_score = 0
        
        for name, model in self.models.items():
            print(f"\nCross-validating {name}...")
            
            cv_results = cross_validate(
                model, X_train, y_train,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                return_train_score=True
            )
            
            # Store results
            self.cv_results[name] = {
                'accuracy': cv_results['test_accuracy'].mean(),
                'accuracy_std': cv_results['test_accuracy'].std(),
                'f1': cv_results['test_f1'].mean(),
                'f1_std': cv_results['test_f1'].std(),
                'roc_auc': cv_results['test_roc_auc'].mean(),
                'roc_auc_std': cv_results['test_roc_auc'].std()
            }
            
            # Print results
            print(f"  Accuracy: {self.cv_results[name]['accuracy']:.4f} "
                  f"(+/- {self.cv_results[name]['accuracy_std']:.4f})")
            print(f"  F1 Score: {self.cv_results[name]['f1']:.4f} "
                  f"(+/- {self.cv_results[name]['f1_std']:.4f})")
            print(f"  ROC-AUC: {self.cv_results[name]['roc_auc']:.4f} "
                  f"(+/- {self.cv_results[name]['roc_auc_std']:.4f})")
            
            # Track best model
            if self.cv_results[name]['roc_auc'] > best_score:
                best_score = self.cv_results[name]['roc_auc']
                self.best_model_name = name
        
        print(f"\nBest model (by ROC-AUC): {self.best_model_name} "
              f"({best_score:.4f})")
        
        return self.cv_results
    
    def train_final_models(self, X_train, y_train):
        """
        Train models on full training set
        """
        print("\n" + "="*70)
        print("TRAINING FINAL MODELS")
        print("="*70)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            print(f"  {name} training complete")
            
            # Store best model
            if name == self.best_model_name:
                self.best_model = model
        
        print("\nAll models trained successfully")
    
    def save_models(self):
        """
        Save trained models to disk
        """
        print("\nSaving models...")
        
        for name, model in self.models.items():
            # Create safe filename
            filename = name.lower().replace(' ', '_') + '.pkl'
            filepath = os.path.join(config.MODELS_DIR, filename)
            
            joblib.dump(model, filepath)
            print(f"  Saved {name} to {filepath}")
        
        # Save best model separately
        best_model_path = os.path.join(config.MODELS_DIR, 'best_model.pkl')
        joblib.dump(self.best_model, best_model_path)
        
        # Save model metadata
        metadata = {
            'best_model': self.best_model_name,
            'cv_results': self.cv_results
        }
        metadata_path = os.path.join(config.MODELS_DIR, 'model_metadata.pkl')
        joblib.dump(metadata, metadata_path)
        
        print(f"\nBest model saved to {best_model_path}")
    
    def save_cv_results(self):
        """
        Save cross-validation results
        """
        cv_df = pd.DataFrame(self.cv_results).T
        cv_df['model'] = cv_df.index
        cv_df = cv_df.reset_index(drop=True)
        
        output_path = config.MODEL_PERFORMANCE_FILE
        cv_df.to_csv(output_path, index=False)
        print(f"\nCV results saved to {output_path}")
    
    def training_pipeline(self):
        """
        Complete model training pipeline
        """
        print("="*70)
        print("MODEL TRAINING PIPELINE")
        print("="*70)
        
        # Step 1: Load data
        df = self.load_data()
        
        # Step 2: Train/test split
        X_train, X_test, y_train, y_test = self.prepare_train_test_split(df)
        
        # Save train/test data
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        train_df.to_csv(config.TRAIN_DATA_FILE, index=False)
        test_df.to_csv(config.TEST_DATA_FILE, index=False)
        print(f"\nTrain data saved to: {config.TRAIN_DATA_FILE}")
        print(f"Test data saved to: {config.TEST_DATA_FILE}")
        
        # Step 3: Initialize models
        self.initialize_models()
        
        # Step 4: Cross-validation
        self.cross_validate_models(X_train, y_train)
        
        # Step 5: Train final models
        self.train_final_models(X_train, y_train)
        
        # Step 6: Save models and results
        self.save_models()
        self.save_cv_results()
        
        print("\n" + "="*70)
        print("MODEL TRAINING COMPLETE")
        print("="*70)
        
        return X_train, X_test, y_train, y_test


def main():
    """
    Main execution function
    """
    trainer = ModelTrainer()
    X_train, X_test, y_train, y_test = trainer.training_pipeline()
    
    print("\nTraining Summary:")
    print(f"Best Model: {trainer.best_model_name}")
    print("\nCross-Validation Results:")
    cv_df = pd.DataFrame(trainer.cv_results).T
    print(cv_df.to_string())


if __name__ == "__main__":
    main()
