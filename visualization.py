import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import joblib
import config
import os

class Visualizer:
    """
    Creates visualizations for model analysis and results
    """
    
    def __init__(self):
        self.setup_style()
        self.figures_dir = os.path.join(config.RESULTS_DIR, 'figures')
        os.makedirs(self.figures_dir, exist_ok=True)
    
    def setup_style(self):
        """
        Setup plotting style
        """
        sns.set_style('whitegrid')
        plt.rcParams['figure.dpi'] = config.FIGURE_DPI
        plt.rcParams['figure.figsize'] = config.FIGURE_SIZE
    
    def plot_roc_curves(self, predictions_file=None):
        """
        Plot ROC curves for all models
        """
        if predictions_file is None:
            predictions_file = config.PREDICTIONS_FILE
        
        print("Plotting ROC curves...")
        df = pd.read_csv(predictions_file)
        
        y_true = df['y_true']
        
        plt.figure(figsize=(10, 8))
        
        # Plot ROC curve for each model
        model_names = ['Random Forest', 'Gradient Boosting', 
                      'Logistic Regression', 'SVM']
        
        for name in model_names:
            proba_col = f'{name}_proba'
            if proba_col in df.columns:
                fpr, tpr, _ = roc_curve(y_true, df[proba_col])
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, linewidth=2,
                        label=f'{name} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - PDAC Detection Models', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(self.figures_dir, 'roc_curves.png')
        plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  Saved to {output_path}")
        plt.close()
    
    def plot_precision_recall_curves(self, predictions_file=None):
        """
        Plot Precision-Recall curves for all models
        """
        if predictions_file is None:
            predictions_file = config.PREDICTIONS_FILE
        
        print("Plotting Precision-Recall curves...")
        df = pd.read_csv(predictions_file)
        
        y_true = df['y_true']
        
        plt.figure(figsize=(10, 8))
        
        model_names = ['Random Forest', 'Gradient Boosting', 
                      'Logistic Regression', 'SVM']
        
        for name in model_names:
            proba_col = f'{name}_proba'
            if proba_col in df.columns:
                precision, recall, _ = precision_recall_curve(y_true, df[proba_col])
                pr_auc = auc(recall, precision)
                
                plt.plot(recall, precision, linewidth=2,
                        label=f'{name} (AUC = {pr_auc:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves - PDAC Detection Models', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc='lower left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(self.figures_dir, 'precision_recall_curves.png')
        plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  Saved to {output_path}")
        plt.close()
    
    def plot_confusion_matrices(self, predictions_file=None):
        """
        Plot confusion matrices for all models
        """
        if predictions_file is None:
            predictions_file = config.PREDICTIONS_FILE
        
        print("Plotting confusion matrices...")
        df = pd.read_csv(predictions_file)
        
        y_true = df['y_true']
        
        model_names = ['Random Forest', 'Gradient Boosting', 
                      'Logistic Regression', 'SVM']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for idx, name in enumerate(model_names):
            pred_col = f'{name}_pred'
            if pred_col in df.columns:
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_true, df[pred_col])
                
                # Normalize
                cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           ax=axes[idx], cbar=False,
                           xticklabels=['Normal', 'PDAC'],
                           yticklabels=['Normal', 'PDAC'])
                
                axes[idx].set_title(name, fontsize=12, fontweight='bold')
                axes[idx].set_ylabel('True Label', fontsize=10)
                axes[idx].set_xlabel('Predicted Label', fontsize=10)
        
        plt.suptitle('Confusion Matrices - PDAC Detection Models',
                    fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        output_path = os.path.join(self.figures_dir, 'confusion_matrices.png')
        plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  Saved to {output_path}")
        plt.close()
    
    def plot_feature_importance(self, n_features=20):
        """
        Plot feature importance for best model
        """
        print("Plotting feature importance...")
        
        importance_file = config.FEATURE_IMPORTANCE_FILE.replace('.csv', '_model.csv')
        
        if not os.path.exists(importance_file):
            print(f"  Feature importance file not found: {importance_file}")
            return
        
        df = pd.read_csv(importance_file)
        df = df.head(n_features)
        
        plt.figure(figsize=(12, 8))
        
        plt.barh(range(len(df)), df['importance'], color='steelblue', alpha=0.8)
        plt.yticks(range(len(df)), df['feature'])
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title(f'Top {n_features} Most Important Features', 
                 fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(self.figures_dir, 'feature_importance.png')
        plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  Saved to {output_path}")
        plt.close()
    
    def plot_model_comparison(self):
        """
        Plot model comparison bar chart
        """
        print("Plotting model comparison...")
        
        metrics_file = config.MODEL_PERFORMANCE_FILE.replace('.csv', '_test.csv')
        
        if not os.path.exists(metrics_file):
            print(f"  Metrics file not found: {metrics_file}")
            return
        
        df = pd.read_csv(metrics_file)
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        x = np.arange(len(df))
        width = 0.15
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for idx, metric in enumerate(metrics):
            offset = width * (idx - 2)
            ax.bar(x + offset, df[metric], width, 
                  label=metric.upper().replace('_', '-'),
                  color=colors[idx], alpha=0.8)
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df['model'], rotation=15, ha='right')
        ax.legend(loc='lower right', fontsize=10)
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(self.figures_dir, 'model_comparison.png')
        plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  Saved to {output_path}")
        plt.close()
    
    def plot_class_distribution(self):
        """
        Plot class distribution in train and test sets
        """
        print("Plotting class distribution...")
        
        train_df = pd.read_csv(config.TRAIN_DATA_FILE)
        test_df = pd.read_csv(config.TEST_DATA_FILE)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Training set
        train_counts = train_df['is_tumor'].value_counts()
        axes[0].bar(['Normal', 'PDAC'], 
                   [train_counts.get(0, 0), train_counts.get(1, 0)],
                   color=['#2ecc71', '#e74c3c'], alpha=0.8)
        axes[0].set_title('Training Set Distribution', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Count', fontsize=10)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Test set
        test_counts = test_df['is_tumor'].value_counts()
        axes[1].bar(['Normal', 'PDAC'], 
                   [test_counts.get(0, 0), test_counts.get(1, 0)],
                   color=['#2ecc71', '#e74c3c'], alpha=0.8)
        axes[1].set_title('Test Set Distribution', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Count', fontsize=10)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.suptitle('Class Distribution in Datasets', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = os.path.join(self.figures_dir, 'class_distribution.png')
        plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  Saved to {output_path}")
        plt.close()
    
    def plot_prediction_distributions(self, predictions_file=None):
        """
        Plot probability distribution for predictions
        """
        if predictions_file is None:
            predictions_file = config.PREDICTIONS_FILE
        
        print("Plotting prediction distributions...")
        df = pd.read_csv(predictions_file)
        
        model_names = ['Random Forest', 'Gradient Boosting', 
                      'Logistic Regression', 'SVM']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        
        for idx, name in enumerate(model_names):
            proba_col = f'{name}_proba'
            if proba_col in df.columns:
                # Separate by true class
                normal = df[df['y_true'] == 0][proba_col]
                pdac = df[df['y_true'] == 1][proba_col]
                
                axes[idx].hist(normal, bins=30, alpha=0.6, 
                             label='Normal', color='#2ecc71')
                axes[idx].hist(pdac, bins=30, alpha=0.6, 
                             label='PDAC', color='#e74c3c')
                
                axes[idx].set_xlabel('Predicted Probability', fontsize=10)
                axes[idx].set_ylabel('Frequency', fontsize=10)
                axes[idx].set_title(name, fontsize=11, fontweight='bold')
                axes[idx].legend(fontsize=9)
                axes[idx].grid(alpha=0.3)
        
        plt.suptitle('Prediction Probability Distributions', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = os.path.join(self.figures_dir, 'prediction_distributions.png')
        plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  Saved to {output_path}")
        plt.close()
    
    def generate_all_plots(self):
        """
        Generate all visualizations
        """
        print("="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        
        self.plot_class_distribution()
        self.plot_roc_curves()
        self.plot_precision_recall_curves()
        self.plot_confusion_matrices()
        self.plot_feature_importance()
        self.plot_model_comparison()
        self.plot_prediction_distributions()
        
        print("\n" + "="*70)
        print("VISUALIZATION COMPLETE")
        print(f"All figures saved to: {self.figures_dir}")
        print("="*70)


def main():
    """
    Main execution function
    """
    visualizer = Visualizer()
    visualizer.generate_all_plots()


if __name__ == "__main__":
    main()
