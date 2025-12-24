# Simple Spam Detector

An AI-powered email and message classification system built with Machine Learning. This project uses Natural Language Processing (NLP) and Naive Bayes classifier to detect spam messages with high accuracy.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3.0-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Features

- 🤖 **Machine Learning Powered**: Uses Multinomial Naive Bayes with TF-IDF vectorization
- 🎯 **High Accuracy**: Achieves excellent performance on spam detection
- 🌐 **Web Interface**: Beautiful and interactive web UI built with Flask
- 📊 **Detailed Analytics**: Provides confidence scores and probability distributions
- 🚀 **Easy to Use**: Simple command-line interface and web application
- 📈 **Model Evaluation**: Comprehensive evaluation with confusion matrix and ROC curve
- 🔄 **Batch Prediction**: Support for multiple message predictions

## Project Structure

```
simple_spam_detector/
├── app/
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css
│   │   └── js/
│   │       └── animations.js
│   ├── templates/
│   │   └── index.html
│   └── app.py
├── data/
│   ├── raw/
│   │   └── spam.csv (not included - see Dataset section)
│   └── processed/
│       ├── X_train.pkl
│       ├── X_test.pkl
│       ├── y_train.pkl
│       └── y_test.pkl
├── model/
│   ├── spam_model.pkl
│   └── vectorizer.pkl
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── run_all.py
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/simple_spam_detector.git
   cd simple_spam_detector
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data** (automatic on first run)
   ```python
   python -c "import nltk; nltk.download('stopwords')"
   ```

## Dataset

This project requires a spam dataset in CSV format. The dataset should have two columns:
- `Category`: Label (spam/ham)
- `Message`: The message text

### Dataset Format Example

```csv
Category,Message
spam,"Congratulations! You've won $1000. Click here now!"
ham,"Hey, are we still meeting for lunch tomorrow?"
spam,"URGENT: Your account has been compromised!"
ham,"Thanks for dinner last night!"
```

### Where to Get the Dataset

You can use publicly available spam datasets such as:
- [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) from Kaggle
- [Enron Email Dataset](http://www.aueb.gr/users/ion/data/enron-spam/)
- [SpamAssassin Public Corpus](https://spamassassin.apache.org/old/publiccorpus/)

### Setup Dataset

1. Download your preferred spam dataset
2. Place the CSV file in the `data/raw/` folder
3. Rename it to `spam.csv` or update the path in `src/preprocess.py`

## Usage

### Quick Start (Recommended)

Use the automated script to run the complete pipeline:

```bash
python run_all.py
```

This will show a menu with options:
1. Run FULL PIPELINE (Preprocess → Train → Evaluate)
2. Run PREPROCESS only
3. Run TRAIN only
4. Run EVALUATE only
5. Run WEB APP only
6. Run TEST PREDICTION

### Manual Step-by-Step

#### 1. Data Preprocessing

Clean and prepare the dataset:

```bash
python src/preprocess.py
```

This will:
- Load the raw dataset
- Clean text (remove URLs, special characters, etc.)
- Apply stemming and remove stopwords
- Split into train/test sets (80/20)
- Save processed data to `data/processed/`

#### 2. Model Training

Train the spam detection model:

```bash
python src/train.py
```

This will:
- Load preprocessed data
- Convert text to TF-IDF features
- Train Naive Bayes classifier
- Save model and vectorizer to `model/`

#### 3. Model Evaluation

Evaluate model performance:

```bash
python src/evaluate.py
```

This will:
- Load test data and trained model
- Calculate metrics (accuracy, precision, recall, F1)
- Generate confusion matrix
- Plot ROC curve
- Save visualizations as PNG files

#### 4. Run Web Application

Start the Flask web server:

```bash
cd app
python app.py
```

Then open your browser and navigate to:
```
http://localhost:5000
```

#### 5. Command-Line Prediction

Test predictions from the command line:

```bash
python src/predict.py
```

## Model Details

### Text Preprocessing

- **Lowercasing**: Convert all text to lowercase
- **URL Removal**: Remove HTTP/HTTPS URLs
- **Email Removal**: Remove email addresses
- **Special Characters**: Remove non-alphabetic characters
- **Stopword Removal**: Remove common English stopwords
- **Stemming**: Apply Porter Stemmer algorithm

### Feature Extraction

- **TF-IDF Vectorization**
  - Max features: 3000
  - N-gram range: (1, 2)
  - Min document frequency: 2
  - Max document frequency: 0.95

### Classification Algorithm

- **Multinomial Naive Bayes**
  - Alpha (smoothing parameter): 0.1
  - Suitable for text classification
  - Fast training and prediction
  - Probabilistic output

## API Endpoints

### `/predict` (POST)

Classify a single message.

**Request:**
```json
{
  "text": "Congratulations! You've won a prize!"
}
```

**Response:**
```json
{
  "success": true,
  "prediction": "Spam",
  "is_spam": true,
  "confidence": 95.67,
  "spam_probability": 95.67,
  "ham_probability": 4.33
}
```

### `/batch-predict` (POST)

Classify multiple messages.

**Request:**
```json
{
  "texts": [
    "Message 1",
    "Message 2"
  ]
}
```

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "prediction": "Spam",
      "confidence": 92.5,
      ...
    },
    ...
  ]
}
```

## Performance Metrics

Expected performance on standard spam datasets:

- **Accuracy**: ~96-98%
- **Precision**: ~95-97%
- **Recall**: ~94-96%
- **F1 Score**: ~95-97%
- **ROC AUC**: ~98-99%

*Note: Actual performance depends on the dataset used.*

## Web Interface Features

- 🎨 Modern and responsive design
- ⚡ Real-time prediction
- 📊 Confidence meters and probability bars
- 💡 Example messages for testing
- ⚠️ Warning indicators for spam messages
- 🎭 Animated UI elements
- 📱 Mobile-friendly

## Customization

### Change Machine Learning Model

Edit `src/train.py` and modify the `model_type` parameter:

```python
trainer.train_pipeline(model_type='logistic_regression')
# Options: 'naive_bayes', 'logistic_regression', 'svm', 'random_forest'
```

### Adjust TF-IDF Parameters

Modify vectorizer settings in `src/train.py`:

```python
self.vectorizer = TfidfVectorizer(
    max_features=5000,  # Increase feature count
    ngram_range=(1, 3),  # Use trigrams
    min_df=3,
    max_df=0.90
)
```

### Modify Train/Test Split

Change the split ratio in `src/preprocess.py`:

```python
test_size=0.3  # 30% for testing, 70% for training
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Create a Pull Request

## Acknowledgments

- Dataset providers and Kaggle community
- scikit-learn for machine learning tools
- Flask for web framework
- NLTK for natural language processing

## Support

If you find this project helpful, please give it a ⭐️!

For issues and questions, please open an issue on GitHub.


## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'flask'`
- **Solution**: Install dependencies with `pip install -r requirements.txt`

**Issue**: `FileNotFoundError: spam.csv not found`
- **Solution**: Place your dataset in `data/raw/spam.csv`

**Issue**: NLTK stopwords not found
- **Solution**: Run `python -c "import nltk; nltk.download('stopwords')"`

**Issue**: Port 5000 already in use
- **Solution**: Change port in `app/app.py`: `app.run(port=5001)`

---

## Author

**Candra**
