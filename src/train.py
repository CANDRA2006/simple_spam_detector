import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from src.preprocess import clean_text

# load dataset
df = pd.read_csv("data/spam.csv")

# asumsi kolom: label, message
df['message'] = df['message'].apply(clean_text)

# ubah label ke angka
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# split data
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# model
model = MultinomialNB()
model.fit(X_train, y_train)

# evaluasi
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# simpan model + vectorizer
with open("model/spam_model.pkl", "wb") as f:
    pickle.dump((model, vectorizer), f)

print("Model saved to model/spam_model.pkl")
