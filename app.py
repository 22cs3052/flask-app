# task3_end_to_end_project/app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import logging
from model import HousePricePredictor

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the trained model
model_path = 'house_price_model.pkl'
if os.path.exists(model_path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.warning("Training new model...")
        predictor = HousePricePredictor()
        model = predictor.train_model()
        logger.info("New model trained and ready")
else:
    logger.warning("No trained model found. Training new model...")
    predictor = HousePricePredictor()
    model = predictor.train_model()
    logger.info("New model trained and ready")

@app.route('/')
def home():
    """Home page with prediction form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction based on input features"""
    try:
        # Get data from form
        data = request.json if request.is_json else request.form
        
        features = {
            'bedrooms': float(data.get('bedrooms', 3)),
            'bathrooms': float(data.get('bathrooms', 2)),
            'sqft_living': float(data.get('sqft_living', 2000)),
            'sqft_lot': float(data.get('sqft_lot', 5000)),
            'floors': float(data.get('floors', 1)),
            'waterfront': 1 if data.get('waterfront') == 'yes' else 0,
            'view': int(data.get('view', 0)),
            'condition': int(data.get('condition', 3)),
            'grade': int(data.get('grade', 7)),
            'yr_built': int(data.get('yr_built', 1990)),
            'yr_renovated': int(data.get('yr_renovated', 0)),
            'zipcode': int(data.get('zipcode', 98178))
        }
        
        # Create DataFrame for prediction
        input_df = pd.DataFrame([features])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Log the prediction
        logger.info(f"Prediction made: ${prediction:,.2f} for features: {features}")
        
        response = {
            'success': True,
            'prediction': f"${prediction:,.2f}",
            'features': features,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/model_info')
def model_info():
    """Get model information"""
    try:
        if hasattr(model, 'feature_names'):
            features = model.feature_names
        else:
            features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
                       'waterfront', 'view', 'condition', 'grade', 'yr_built', 
                       'yr_renovated', 'zipcode']
        
        return jsonify({
            'model_type': type(model).__name__,
            'features': features,
            'n_features': len(features)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n🏠 House Price Prediction Flask App")
    print("=" * 50)
    print("🚀 Starting Flask server...")
    print("📊 Model ready for predictions!")
    print("🌐 Open your browser to: http://127.0.0.1:5000/")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)