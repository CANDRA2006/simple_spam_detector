import pickle
import joblib
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class ModelEvaluator:
    def __init__(self, model_dir='model'):
        self.model = joblib.load(os.path.join(model_dir, 'spam_model.pkl'))
        self.vectorizer = joblib.load(os.path.join(model_dir, 'vectorizer.pkl'))
    
    def load_test_data(self, data_dir='data/processed'):
        """Load test data"""
        with open(os.path.join(data_dir, 'X_test.pkl'), 'rb') as f:
            X_test = pickle.load(f)
        
        with open(os.path.join(data_dir, 'y_test.pkl'), 'rb') as f:
            y_test = pickle.load(f)
        
        return X_test, y_test
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        # Vectorize test data
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Predictions
        y_pred = self.model.predict(X_test_vec)
        y_pred_proba = self.model.predict_proba(X_test_vec)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print("=" * 50)
        print("MODEL EVALUATION RESULTS")
        print("=" * 50)
        print(f"\nAccuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
        print(f"ROC AUC:   {roc_auc:.4f} ({roc_auc*100:.2f}%)")
        
        print("\n" + "=" * 50)
        print("CLASSIFICATION REPORT")
        print("=" * 50)
        print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    def plot_confusion_matrix(self, cm, save_path=None):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Ham', 'Spam'], 
                    yticklabels=['Ham', 'Spam'])
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('Actual Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, y_test, y_pred_proba, roc_auc, save_path=None):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                 label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_full(self, data_dir='data/processed', save_plots=False):
        """Complete evaluation pipeline"""
        X_test, y_test = self.load_test_data(data_dir)
        results = self.evaluate(X_test, y_test)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(
            results['confusion_matrix'],
            save_path='confusion_matrix.png' if save_plots else None
        )
        
        # Plot ROC curve
        self.plot_roc_curve(
            y_test, 
            results['y_pred_proba'], 
            results['roc_auc'],
            save_path='roc_curve.png' if save_plots else None
        )
        
        return results

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_full(save_plots=True)