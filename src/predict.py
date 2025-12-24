import joblib
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

class SpamPredictor:
    def __init__(self, model_dir='model'):
        """Initialize predictor with trained model"""
        self.model = joblib.load(os.path.join(model_dir, 'spam_model.pkl'))
        self.vectorizer = joblib.load(os.path.join(model_dir, 'vectorizer.pkl'))
        
        self.stemmer = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Clean and preprocess text"""
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
    
    def predict(self, text):
        """Predict if text is spam or not"""
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Vectorize
        text_vec = self.vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = self.model.predict(text_vec)[0]
        probability = self.model.predict_proba(text_vec)[0]
        
        result = {
            'text': text,
            'cleaned_text': cleaned_text,
            'prediction': 'Spam' if prediction == 1 else 'Ham',
            'is_spam': bool(prediction),
            'spam_probability': float(probability[1]),
            'ham_probability': float(probability[0]),
            'confidence': float(max(probability))
        }
        
        return result
    
    def predict_batch(self, texts):
        """Predict multiple texts"""
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results

if __name__ == "__main__":
    # Test the predictor
    predictor = SpamPredictor()
    
    test_messages = [
        "Congratulations! You've won a $1000 gift card. Click here to claim now!",
        "Hey, are we still meeting for lunch tomorrow?",
        "URGENT: Your account has been compromised. Reset password immediately!",
        "Thanks for the dinner last night, had a great time!",
        "FREE entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005"
    ]
    
    print("=" * 70)
    print("SPAM DETECTION PREDICTIONS")
    print("=" * 70)
    
    for msg in test_messages:
        result = predictor.predict(msg)
        print(f"\nMessage: {msg[:60]}...")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']*100:.2f}%")
        print(f"Spam Probability: {result['spam_probability']*100:.2f}%")
        print("-" * 70)