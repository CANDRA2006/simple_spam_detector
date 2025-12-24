from flask import Flask, render_template, request, jsonify
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import SpamPredictor

app = Flask(__name__)

# Initialize predictor
try:
    predictor = SpamPredictor(model_dir='../model')
except:
    predictor = SpamPredictor(model_dir='model')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text or text.strip() == '':
            return jsonify({
                'error': 'Please enter a message to check'
            }), 400
        
        # Get prediction
        result = predictor.predict(text)
        
        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'is_spam': result['is_spam'],
            'confidence': round(result['confidence'] * 100, 2),
            'spam_probability': round(result['spam_probability'] * 100, 2),
            'ham_probability': round(result['ham_probability'] * 100, 2)
        })
    
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}'
        }), 500

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts:
            return jsonify({
                'error': 'Please provide messages to check'
            }), 400
        
        # Get predictions
        results = predictor.predict_batch(texts)
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)