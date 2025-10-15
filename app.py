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
from werkzeug.utils import secure_filename
import tempfile
import os
from src.ml.train_win_probability_model import load_and_prepare_data
from src.utils.cache_utils import load_demo
from src.core.analysis import get_kill_death_analysis
from awpy import Demo

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

# Storage for uploaded demos (persistent across restarts)
uploaded_demos = {}  # demo_id -> demo_data
UPLOAD_FOLDER = PROJECT_ROOT / "cache" / "demo_uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
DEMO_METADATA_FILE = UPLOAD_FOLDER / "demos_metadata.json"
ALLOWED_EXTENSIONS = {'dem'}

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_demo_metadata():
    """Save demo metadata to disk"""
    metadata = {}
    for demo_id, demo_data in uploaded_demos.items():
        metadata[demo_id] = {
            'filename': demo_data['filename'],
            'players': demo_data['players'],
            'path': str(demo_data['path'])
        }
    
    with open(DEMO_METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)

def load_demo_metadata():
    """Load demo metadata from disk on startup"""
    global uploaded_demos
    
    if not DEMO_METADATA_FILE.exists():
        return
    
    try:
        with open(DEMO_METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        
        print(f"📂 Loading {len(metadata)} cached demos...")
        
        for demo_id, demo_info in metadata.items():
            demo_path = Path(demo_info['path'])
            
            # Check if demo file still exists
            if not demo_path.exists():
                print(f"⚠️ Demo file not found: {demo_path}")
                continue
            
            # Store minimal metadata (don't parse demo yet for performance)
            uploaded_demos[demo_id] = {
                'filename': demo_info['filename'],
                'players': demo_info['players'],
                'path': demo_path,
                'demo': None  # Will be loaded on demand
            }
        
        print(f"✅ Loaded {len(uploaded_demos)} demos from cache")
        
    except Exception as e:
        print(f"❌ Error loading demo metadata: {e}")
        uploaded_demos = {}

def get_or_parse_demo(demo_id):
    """Get demo object, parsing it if not already loaded"""
    if demo_id not in uploaded_demos:
        return None
    
    demo_data = uploaded_demos[demo_id]
    
    # If demo not parsed yet, parse it now
    if demo_data['demo'] is None:
        try:
            print(f"🔄 Parsing demo: {demo_data['filename']}")
            dem = Demo(str(demo_data['path']))
            dem.parse(player_props=['armor_value', 'has_helmet', 'has_defuser', 'inventory'])
            demo_data['demo'] = dem
            print(f"✅ Demo parsed: {demo_data['filename']}")
        except Exception as e:
            print(f"❌ Error parsing demo: {e}")
            raise
    
    return demo_data['demo']



@app.route('/api/slice_dataset', methods=['POST'])
def slice_dataset():
    data = request.json
    features_with_bins = data.get('features_with_bins', {})

    mask = pd.Series(True, index=dataset_df.index)
    for key, cfg in features_with_bins.items():
        v, bs = cfg['value'], cfg['bin_size']
        if bs < 0:
            continue
        mask &= dataset_df[key].between(v - bs, v + bs)

    df_filtered = dataset_df[mask]

    n_samples = len(df_filtered)

    probability = len(df_filtered[df_filtered['winner'] == 'ct']) / n_samples if n_samples > 0 else -1

    return jsonify({
        'ct_win_probability': float(probability),
        't_win_probability': float(1 - probability),
        'n_samples': n_samples
    })
        




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
        # Create display name based on model type
        if model_name.startswith('xgboost_'):
            display_name = f"XGBoost {model_name.replace('xgboost_', '').replace('_', ' ').title()}"
        elif model_name == 'lightgbm':
            display_name = "LightGBM"
        elif model_name == 'random_forest':
            display_name = "Random Forest"
        elif model_name == 'logistic_regression':
            display_name = "Logistic Regression"
        elif model_name == 'neural_network':
            display_name = "Neural Network"
        elif model_name == 'ensemble':
            display_name = "Ensemble Model"
        else:
            display_name = model_name.replace('_', ' ').title()
        
        # Get description from config or create default
        config = model_data.get('config', {})
        if config and 'description' in config:
            description = config['description']
        else:
            description = f"{display_name} model for win probability prediction"
        
        models_info[model_name] = {
            'name': model_name,
            'display_name': display_name,
            'accuracy': model_data.get('accuracy', 0),
            'auc': model_data.get('auc', 0),
            'log_loss': model_data.get('log_loss', 0),
            'features': model_data.get('feature_columns', []),
            'feature_count': len(model_data.get('feature_columns', [])),
            'description': description,
            'rank': model_data.get('rank', 999)  # Default rank for sorting
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

@app.route('/api/demo/upload', methods=['POST'])
def upload_demo():
    """Upload and parse a .dem file, return list of players"""
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type. Only .dem files are allowed'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        demo_id = f"{os.urandom(8).hex()}_{filename}"
        demo_path = UPLOAD_FOLDER / demo_id
        file.save(demo_path)
        
        print(f"📁 Uploaded demo: {filename}")
        
        # Parse demo
        try:
            dem = Demo(str(demo_path))
            dem.parse(player_props=['armor_value', 'has_helmet', 'has_defuser', 'inventory'])
            print(f"✅ Demo parsed successfully: {filename}")
        except Exception as e:
            print(f"❌ Error parsing demo: {e}")
            demo_path.unlink()  # Clean up file
            return jsonify({'success': False, 'error': f'Failed to parse demo: {str(e)}'}), 500
        
        # Extract unique player names
        import polars as pl
        ticks_df = dem.ticks if isinstance(dem.ticks, pl.DataFrame) else pl.from_pandas(dem.ticks)
        players = sorted(ticks_df['name'].unique().to_list())
        
        # Store demo data persistently
        uploaded_demos[demo_id] = {
            'filename': filename,
            'demo': dem,
            'players': players,
            'path': demo_path
        }
        
        # Save metadata to disk
        save_demo_metadata()
        
        return jsonify({
            'success': True,
            'demo_id': demo_id,
            'filename': filename,
            'players': players,
            'player_count': len(players)
        })
        
    except Exception as e:
        print(f"❌ Upload error: {e}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/demo/<demo_id>/analysis', methods=['GET'])
def analyze_all_players(demo_id):
    """Get kill/death analysis for all players in an uploaded demo"""
    global loaded_models
    try:
        # Check if demo exists
        if demo_id not in uploaded_demos:
            return jsonify({'success': False, 'error': 'Demo not found'}), 404
        
        demo_data = uploaded_demos[demo_id]
        
        print(f"🔍 Analyzing all players in {demo_data['filename']}")
        
        # Get or parse demo (lazy loading)
        dem = get_or_parse_demo(demo_id)
        if dem is None:
            return jsonify({'success': False, 'error': 'Failed to load demo'}), 500
        
        # Run analysis for all players at once
        all_players_analysis = get_kill_death_analysis(dem, loaded_models['xgboost_hltv'], debug=False)
        
        if all_players_analysis is None or len(all_players_analysis) == 0:
            return jsonify({'success': False, 'error': 'No analysis data generated'}), 500
        
        # Convert numpy/pandas types to Python native types for JSON serialization
        def convert_to_native_types(obj):
            """Recursively convert numpy and pandas types to Python native types"""
            if isinstance(obj, dict):
                return {k: convert_to_native_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif pd.isna(obj):
                return None
            else:
                return obj
        
        # Convert analysis results
        all_players_analysis_serializable = convert_to_native_types(all_players_analysis)
        
        return jsonify({
            'success': True,
            'demo_filename': demo_data['filename'],
            'analysis': all_players_analysis_serializable,
            'players': demo_data['players']
        })
        
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/demo/<demo_id>', methods=['DELETE'])
def delete_demo(demo_id):
    """Delete an uploaded demo"""
    try:
        if demo_id not in uploaded_demos:
            return jsonify({'success': False, 'error': 'Demo not found'}), 404
        
        demo_data = uploaded_demos[demo_id]
        
        # Delete file
        if demo_data['path'].exists():
            demo_data['path'].unlink()
        
        # Remove from memory
        del uploaded_demos[demo_id]
        
        # Update metadata file
        save_demo_metadata()
        
        return jsonify({'success': True, 'message': 'Demo deleted successfully'})
        
    except Exception as e:
        print(f"❌ Delete error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("🎮 CS2 Win Probability Prediction App")
    print("=" * 50)
    
    # Load cached demos from disk
    load_demo_metadata()
    
    # Load model summary on startup
    load_model_summary()
    
    models_info = get_model_info()
    print(f"📊 Available models: {len(models_info)}")
    for model_name, info in models_info.items():
        print(f"   - {info['display_name']}: {info['feature_count']} features, AUC={info['auc']:.3f}")
    
    _, _, _, dataset_df = load_and_prepare_data(data_file=None, check_data=False)
    print("\n🚀 Starting Flask app on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)