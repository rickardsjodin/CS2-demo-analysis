"""
CS2 Win Probability Prediction Flask App
A web interface for predicting CT win probability using trained XGBoost models
"""

import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from pathlib import Path
import traceback

app = Flask(__name__)

# Enable CORS for all routes
CORS(app, origins=['http://localhost:5173', 'http://localhost:3000', 'http://127.0.0.1:5173'])

# Project configuration
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "data" / "models"
TEMPLATES_DIR = PROJECT_ROOT / "templates"
STATIC_DIR = PROJECT_ROOT / "static"

# Create directories if they don't exist
TEMPLATES_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# Global variables for loaded models and features
loaded_models = {}
model_summary = {}

def load_model_summary():
    """Load the model summary containing available models and their info"""
    global model_summary
    summary_path = MODELS_DIR / "model_summary.json"
    
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            model_summary = json.load(f)
        print(f"✅ Loaded model summary with {len(model_summary.get('models', {}))} models")
    else:
        print("⚠️ Model summary not found. Please train models first.")
        model_summary = {"models": {}, "best_model": None}

def get_model_info():
    """Get information about available models"""
    models_info = {}
    
    for model_name, model_data in model_summary.get('models', {}).items():
        if model_name.startswith('xgboost_'):
            models_info[model_name] = {
                'name': model_name,
                'display_name': model_name.replace('xgboost_', '').replace('_', ' ').title(),
                'accuracy': model_data.get('accuracy', 0),
                'auc': model_data.get('auc', 0),
                'features': model_data.get('feature_columns', []),
                'feature_count': len(model_data.get('feature_columns', [])),
                'description': model_data.get('config', {}).get('description', 'XGBoost model')
            }
    
    return models_info

def load_model(model_name):
    """Load a specific model from disk"""
    global loaded_models
    
    if model_name in loaded_models:
        return loaded_models[model_name]
    
    model_path = MODELS_DIR / f"ct_win_probability_{model_name}.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        model_data = joblib.load(model_path)
        loaded_models[model_name] = model_data
        print(f"✅ Loaded model: {model_name}")
        return model_data
    except Exception as e:
        print(f"❌ Error loading model {model_name}: {e}")
        raise

def get_feature_defaults(features):
    """Get reasonable default values for features"""
    defaults = {}
    player_side_i = 0
    for feature in features:
        if 'time_left' in feature.lower():
            if 'bomb' in feature.lower():
                defaults[feature] = 35.0  # Bomb timer
            else:
                defaults[feature] = 75.0  # Round timer
        elif 'alive' in feature.lower():
            if 'ct' in feature.lower():
                defaults[feature] = 5
            else:  # ts_alive
                defaults[feature] = 5
        elif feature.lower() == 'bomb_planted':
            defaults[feature] = 0  # Boolean: 0 = False, 1 = True
        elif 'hp_' in feature.lower():
            defaults[feature] = 500  # Total HP for team
        elif 'main_weapons' in feature.lower():
            defaults[feature] = 3  # Number of main weapons
        elif 'defusers' in feature.lower():
            defaults[feature] = 2  # Number of defuse kits
        elif 'armor' in feature.lower() and 'player_' not in feature.lower():
            defaults[feature] = 4  # Number of players with armor
        elif 'helmets' in feature.lower():
            defaults[feature] = 4  # Number of players with helmets
        elif 'smokes' in feature.lower():
            defaults[feature] = 2  # Number of smoke grenades
        elif 'flashes' in feature.lower():
            defaults[feature] = 3  # Number of flash grenades
        elif 'he_nades' in feature.lower():
            defaults[feature] = 2  # Number of HE grenades
        elif 'molotovs' in feature.lower():
            defaults[feature] = 1  # Number of molotovs/incendiaries
        elif 'player_' in feature.lower() and 'health' in feature.lower():
            defaults[feature] = 80  # Individual player health
        elif 'player_' in feature.lower() and 'best_weapon_tier' in feature.lower():
            defaults[feature] = 6  # Weapon tier (rifles = 6)
        elif 'player_' in feature.lower() and 'has_defuser' in feature.lower():
            defaults[feature] = 0  # Boolean
        elif 'player_' in feature.lower() and 'has_helmet' in feature.lower():
            defaults[feature] = 1  # Boolean
        elif 'player_' in feature.lower() and 'armor' in feature.lower():
            defaults[feature] = 1  # Armor value
        elif 'player_' in feature.lower() and 'side' in feature.lower():
            defaults[feature] = 0 if player_side_i < 5 else 1  # 0 = CT, 1 = T
            player_side_i += 1
        else:
            defaults[feature] = 0  # Default to 0 for unknown features
    
    return defaults

def get_feature_constraints(feature):
    """Get input constraints for features"""
    constraints = {
        'type': 'number',
        'min': 0,
        'max': 100,
        'step': 1
    }
    
    if 'time_left' in feature.lower():
        if 'bomb' in feature.lower():
            constraints.update({'min': 0, 'max': 40, 'step': 1})
        else:
            constraints.update({'min': 0, 'max': 115, 'step': 1})
    elif 'alive' in feature.lower():
        constraints.update({'min': 0, 'max': 5, 'step': 1})
    elif feature.lower() == 'bomb_planted':
        constraints.update({'type': 'checkbox'})
    elif 'hp_' in feature.lower():
        constraints.update({'min': 0, 'max': 500, 'step': 10})
    elif 'main_weapons' in feature.lower():
        constraints.update({'min': 0, 'max': 5, 'step': 1})
    elif 'defusers' in feature.lower():
        constraints.update({'min': 0, 'max': 5, 'step': 1})
    elif 'armor' in feature.lower() and 'player_' not in feature.lower():
        constraints.update({'min': 0, 'max': 5, 'step': 1})
    elif 'helmets' in feature.lower():
        constraints.update({'min': 0, 'max': 5, 'step': 1})
    elif any(nade in feature.lower() for nade in ['smokes', 'flashes', 'he_nades', 'molotovs']):
        constraints.update({'min': 0, 'max': 10, 'step': 1})
    elif 'player_' in feature.lower() and 'health' in feature.lower():
        constraints.update({'min': 0, 'max': 100, 'step': 1})
    elif 'player_' in feature.lower() and 'best_weapon_tier' in feature.lower():
        constraints.update({'min': 0, 'max': 8, 'step': 1})
    elif 'player_' in feature.lower() and ('has_defuser' in feature.lower() or 'has_helmet' in feature.lower()):
        constraints.update({'type': 'checkbox'})
    elif 'player_' in feature.lower() and 'armor' in feature.lower():
        constraints.update({'type': 'checkbox'})
    elif 'player_' in feature.lower() and 'side' in feature.lower():
        constraints.update({'type': 'select', 'options': [('0', 'CT'), ('1', 'T'), ('-1', 'Dead')]})
    
    return constraints

@app.route('/')
def index():
    """Main page with model selection and feature inputs"""
    models_info = get_model_info()
    return render_template('index.html', models=models_info)

@app.route('/api/model/<model_name>/features')
def get_model_features(model_name):
    """Get features for a specific model"""
    try:
        model_data = load_model(model_name)
        features = model_data['feature_columns']
        defaults = get_feature_defaults(features)
        
        feature_data = []
        for feature in features:
            constraints = get_feature_constraints(feature)
            feature_data.append({
                'name': feature,
                'display_name': feature.replace('_', ' ').title(),
                'default': defaults.get(feature, 0),
                'constraints': constraints
            })
        
        return jsonify({
            'success': True,
            'features': feature_data,
            'feature_count': len(features)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction using selected model and features"""
    try:
        data = request.json
        model_name = data.get('model_name')
        features = data.get('features', {})
        
        if not model_name:
            return jsonify({'success': False, 'error': 'Model name is required'}), 400
        
        # Load the model
        model_data = load_model(model_name)
        model = model_data['model']
        scaler = model_data.get('scaler')
        feature_columns = model_data['feature_columns']
        
        # Prepare feature vector
        feature_vector = []
        for feature_name in feature_columns:
            value = features.get(feature_name, 0)
            
            # Handle boolean features
            if isinstance(value, str) and value.lower() in ['true', 'false']:
                value = value.lower() == 'true'
            elif isinstance(value, bool):
                value = int(value)
            else:
                value = float(value)
            
            feature_vector.append(value)
        
        # Convert to numpy array and reshape for prediction
        X = np.array(feature_vector).reshape(1, -1)
        
        # Apply scaling if needed
        if scaler is not None:
            X = scaler.transform(X)
        
        # Make prediction
        probability = model.predict_proba(X)[0, 1]  # Probability of CT win
        prediction = int(probability > 0.5)
        
        # Get model info for context
        model_info = model_summary.get('models', {}).get(model_name, {})
        
        return jsonify({
            'success': True,
            'prediction': {
                'ct_win_probability': float(probability),
                't_win_probability': float(1 - probability),
                'predicted_winner': 'CT' if prediction == 1 else 'T',
                'confidence': float(max(probability, 1 - probability))
            },
            'model_info': {
                'name': model_name,
                'accuracy': model_info.get('accuracy', 0),
                'auc': model_info.get('auc', 0),
                'feature_count': len(feature_columns)
            }
        })
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models')
def get_models():
    """Get list of available models"""
    models_info = get_model_info()
    return jsonify({
        'success': True,
        'models': list(models_info.values())
    })

if __name__ == '__main__':
    print("🎮 CS2 Win Probability Prediction App")
    print("=" * 50)
    
    # Load model summary on startup
    load_model_summary()
    
    models_info = get_model_info()
    print(f"📊 Available models: {len(models_info)}")
    for model_name, info in models_info.items():
        print(f"   - {info['display_name']}: {info['feature_count']} features, AUC={info['auc']:.3f}")
    
    print("\n🚀 Starting Flask app on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)