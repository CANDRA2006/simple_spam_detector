import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib

class SpamDetectorTrainer:
    def __init__(self):
        self.vectorizer = None
        self.model = None
    
    def load_data(self, data_dir):
        """Load preprocessed data"""
        print("Loading preprocessed data...")
        
        with open(os.path.join(data_dir, 'X_train.pkl'), 'rb') as f:
            X_train = pickle.load(f)
        
        with open(os.path.join(data_dir, 'X_test.pkl'), 'rb') as f:
            X_test = pickle.load(f)
        
        with open(os.path.join(data_dir, 'y_train.pkl'), 'rb') as f:
            y_train = pickle.load(f)
        
        with open(os.path.join(data_dir, 'y_test.pkl'), 'rb') as f:
            y_test = pickle.load(f)
        
        return X_train, X_test, y_train, y_test
    
    def vectorize_text(self, X_train, X_test):
        """Convert text to TF-IDF features"""
        print("\nVectorizing text using TF-IDF...")
        
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2)
        )
        
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        print(f"Feature matrix shape: {X_train_vec.shape}")
        
        return X_train_vec, X_test_vec
    
    def train_model(self, X_train, y_train, model_type='naive_bayes'):
        """Train the spam detection model"""
        print(f"\nTraining {model_type} model...")
        
        if model_type == 'naive_bayes':
            self.model = MultinomialNB(alpha=0.1)
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(max_iter=1000, C=1.0)
        elif model_type == 'svm':
            self.model = SVC(kernel='linear', C=1.0, probability=True)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.fit(X_train, y_train)
        print("Training complete!")
        
        return self.model
    
    def save_model(self, model_dir):
        """Save trained model and vectorizer"""
        print("\nSaving model and vectorizer...")
        
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(self.model, os.path.join(model_dir, 'spam_model.pkl'))
        joblib.dump(self.vectorizer, os.path.join(model_dir, 'vectorizer.pkl'))
        
        print("Model saved successfully!")
    
    def train_pipeline(self, data_dir='data/processed', model_dir='model', model_type='naive_bayes'):
        """Complete training pipeline"""
        # Load data
        X_train, X_test, y_train, y_test = self.load_data(data_dir)
        
        # Vectorize
        X_train_vec, X_test_vec = self.vectorize_text(X_train, X_test)
        
        # Train
        self.train_model(X_train_vec, y_train, model_type)
        
        # Save
        self.save_model(model_dir)
        
        return X_test_vec, y_test

if __name__ == "__main__":
    trainer = SpamDetectorTrainer()
    X_test_vec, y_test = trainer.train_pipeline(model_type='naive_bayes')