import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
import pickle
import os

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class TextPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        # Convert to string if not already
        text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        words = text.split()
        
        # Remove stopwords and apply stemming
        words = [self.stemmer.stem(word) for word in words if word not in self.stop_words]
        
        return ' '.join(words)
    
    def preprocess_dataset(self, input_path, output_dir, test_size=0.2, random_state=42):
        """Load and preprocess the spam dataset"""
        print("Loading dataset...")
        df = pd.read_csv(input_path, encoding='utf-8')
        
        print(f"Original columns: {df.columns.tolist()}")
        
        # Handle different CSV formats
        # Your dataset has 'Category' and 'Message' columns
        if 'Category' in df.columns and 'Message' in df.columns:
            df = df[['Category', 'Message']]
            df.columns = ['label', 'text']
        elif 'v1' in df.columns and 'v2' in df.columns:
            df = df[['v1', 'v2']]
            df.columns = ['label', 'text']
        elif 'label' not in df.columns or 'text' not in df.columns:
            # Assume first column is label, second is text
            df.columns = ['label', 'text'] + [f'col_{i}' for i in range(len(df.columns) - 2)]
            df = df[['label', 'text']]
        
        # Remove any NaN values
        df = df.dropna()
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Convert labels to lowercase for consistency
        df['label'] = df['label'].str.lower()
        
        print(f"\nDataset size: {len(df)}")
        print(f"Spam messages: {sum(df['label'] == 'spam')}")
        print(f"Ham messages: {sum(df['label'] == 'ham')}")
        
        # Clean text
        print("\nCleaning text...")
        df['cleaned_text'] = df['text'].apply(self.clean_text)
        
        # Convert labels to binary
        df['label'] = df['label'].map({'spam': 1, 'ham': 0})
        
        # Check for any unmapped labels
        if df['label'].isna().any():
            print("\nWarning: Some labels were not mapped correctly!")
            print(df[df['label'].isna()]['label'].value_counts())
            df = df.dropna(subset=['label'])
        
        # Split dataset
        print("\nSplitting dataset...")
        X = df['cleaned_text']
        y = df['label']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save processed data
        print("\nSaving processed data...")
        with open(os.path.join(output_dir, 'X_train.pkl'), 'wb') as f:
            pickle.dump(X_train, f)
        
        with open(os.path.join(output_dir, 'X_test.pkl'), 'wb') as f:
            pickle.dump(X_test, f)
        
        with open(os.path.join(output_dir, 'y_train.pkl'), 'wb') as f:
            pickle.dump(y_train, f)
        
        with open(os.path.join(output_dir, 'y_test.pkl'), 'wb') as f:
            pickle.dump(y_test, f)
        
        print(f"\nTrain set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        print(f"Train spam ratio: {y_train.sum() / len(y_train) * 100:.2f}%")
        print(f"Test spam ratio: {y_test.sum() / len(y_test) * 100:.2f}%")
        print("\nPreprocessing complete!")
        print(f"Files saved in: {output_dir}")
        
        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    
    # Check if file exists
    input_file = 'data/raw/spam.csv'
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found!")
        print("Please place your spam.csv file in the data/raw/ folder")
    else:
        preprocessor.preprocess_dataset(
            input_path=input_file,
            output_dir='data/processed'
        )