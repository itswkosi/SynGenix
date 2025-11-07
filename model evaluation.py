import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
import joblib
import config
import os

class ModelEvaluator:
    """
    Evaluates PDAC detection models
    """
    
    def __init__(self):
        self.models = {}
        self.predictions = {}
        self.metrics = {}
    
    def load_test_data(self):
        """
        Load test dataset
        """
        print(f"Loading test data from {config.TEST_DATA_FILE}...")
        df = pd.read_csv(config.TEST_DATA_FILE)
        
        X_test = df.drop(columns=['is_tumor'])
        y_test = df['is_tumor']
        
        print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        
        return X_test, y_test
    
    def load_models(self):
        """
        Load trained models
        """
        print("\nLoading trained models...")
        
        model_files = {
            'Random Forest': 'random_forest.pkl',
            'Gradient Boosting': 'gradient_boosting.pkl',
            'Logistic Regression': 'logistic_regression.pkl',
            'SVM': 'svm.pkl'
        }
        
        for name, filename in model_files.items():
            filepath = os.path.join(config.MODELS_DIR, filename)
            if os.path.exists(filepath):
                self.models[name] = joblib.load(filepath)
                print(f"  Loaded {name}")
        
        # Load metadata
        metadata_path = os.path.join(config.MODELS_DIR, 'model_metadata.pkl')
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.best_model_name = metadata.get('best_model')
            print(f"\nBest model: {self.best_model_name}")
        
        return self.models
    
    def predict(self, X_test):
        """
        Generate predictions for all models
        """
        print("\nGenerating predictions...")
        
        for name, model in self.models.items():
            self.predictions[name] = {
                'y_pred': model.predict(X_test),
                'y_proba': model.predict_proba(X_test)[:, 1]
            }
            print(f"  {name}: predictions generated")
        
        return self.predictions
    
    def calculate_metrics(self, y_test):
        """
        Calculate evaluation metrics
        """
        print("\nCalculating metrics...")
        
        for name in self.models.keys():
            y_pred = self.predictions[name]['y_pred']
            y_proba = self.predictions[name]['y_proba']
            
            self.metrics[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_proba),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
        
        return self.metrics
    
    def print_evaluation_report(self, y_test):
        """
        Print comprehensive evaluation report
        """
        print("\n" + "="*70)
        print("MODEL EVALUATION REPORT")
        print("="*70)
        
        for name in self.models.keys():
            print(f"\n{name}")
            print("-" * 70)
            
            metrics = self.metrics[name]
            
            print(f"Accuracy:  {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall:    {metrics['recall']:.4f}")
            print(f"F1 Score:  {metrics['f1']:.4f}")
            print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
            
            print("\nConfusion Matrix:")
            cm = metrics['confusion_matrix']
            print(f"  TN: {cm[0,0]:4d}  |  FP: {cm[0,1]:4d}")
            print(f"  FN: {cm[1,0]:4d}  |  TP: {cm[1,1]:4d}")
            
            # Calculate additional metrics
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            print(f"\nSensitivity (TPR): {sensitivity:.4f}")
            print(f"Specificity (TNR): {specificity:.4f}")
            
            # Classification report
            y_pred = self.predictions[name]['y_pred']
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, 
                                       target_names=['Normal', 'PDAC'],
                                       zero_division=0))
    
    def get_feature_importance(self, feature_names, model_name=None):
        """
        Extract feature importance from tree-based models
        """
        if model_name is None:
            model_name = self.best_model_name
        
        model = self.models.get(model_name)
        
        if not hasattr(model, 'feature_importances_'):
            print(f"{model_name} does not have feature_importances_ attribute")
            return None
        
        importance = model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_results(self, X_test, y_test):
        """
        Save evaluation results
        """
        print("\nSaving results...")
        
        # Save metrics
        metrics_df = pd.DataFrame(self.metrics).T
        metrics_df['model'] = metrics_df.index
        metrics_df = metrics_df[['model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']]
        
        output_path = config.MODEL_PERFORMANCE_FILE.replace('.csv', '_test.csv')
        metrics_df.to_csv(output_path, index=False)
        print(f"  Test metrics saved to {output_path}")
        
        # Save predictions
        predictions_df = pd.DataFrame({'y_true': y_test.values})
        
        for name in self.models.keys():
            predictions_df[f'{name}_pred'] = self.predictions[name]['y_pred']
            predictions_df[f'{name}_proba'] = self.predictions[name]['y_proba']
        
        predictions_df.to_csv(config.PREDICTIONS_FILE, index=False)
        print(f"  Predictions saved to {config.PREDICTIONS_FILE}")
        
        # Save feature importance
        feature_names = X_test.columns.tolist()
        importance_df = self.get_feature_importance(feature_names)
        
        if importance_df is not None:
            importance_path = config.FEATURE_IMPORTANCE_FILE.replace('.csv', '_model.csv')
            importance_df.to_csv(importance_path, index=False)
            print(f"  Feature importance saved to {importance_path}")
    
    def compare_models(self):
        """
        Compare all models
        """
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        
        comparison_df = pd.DataFrame({
            name: {
                'Accuracy': self.metrics[name]['accuracy'],
                'Precision': self.metrics[name]['precision'],
                'Recall': self.metrics[name]['recall'],
                'F1-Score': self.metrics[name]['f1'],
                'ROC-AUC': self.metrics[name]['roc_auc']
            }
            for name in self.models.keys()
        }).T
        
        print("\n", comparison_df.to_string())
        
        # Find best model per metric
        print("\n\nBest Model per Metric:")
        print("-" * 70)
        for metric in comparison_df.columns:
            best_model = comparison_df[metric].idxmax()
            best_score = comparison_df[metric].max()
            print(f"{metric:12s}: {best_model:20s} ({best_score:.4f})")
        
        return comparison_df
    
    def evaluation_pipeline(self):
        """
        Complete evaluation pipeline
        """
        print("="*70)
        print("MODEL EVALUATION PIPELINE")
        print("="*70)
        
        # Step 1: Load test data
        X_test, y_test = self.load_test_data()
        
        # Step 2: Load models
        self.load_models()
        
        # Step 3: Generate predictions
        self.predict(X_test)
        
        # Step 4: Calculate metrics
        self.calculate_metrics(y_test)
        
        # Step 5: Print evaluation report
        self.print_evaluation_report(y_test)
        
        # Step 6: Compare models
        comparison = self.compare_models()
        
        # Step 7: Save results
        self.save_results(X_test, y_test)
        
        print("\n" + "="*70)
        print("MODEL EVALUATION COMPLETE")
        print("="*70)
        
        return self.metrics, comparison


def main():
    """
    Main execution function
    """
    evaluator = ModelEvaluator()
    metrics, comparison = evaluator.evaluation_pipeline()
    
    print("\n\nFinal Summary:")
    print("="*70)
    print(f"Best Overall Model: {evaluator.best_model_name}")
    print(f"ROC-AUC Score: {metrics[evaluator.best_model_name]['roc_auc']:.4f}")
    print(f"F1 Score: {metrics[evaluator.best_model_name]['f1']:.4f}")


if __name__ == "__main__":
    main()
