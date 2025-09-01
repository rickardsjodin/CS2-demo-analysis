"""
CS2 Win Probability ML Model Training
Trains a model to predict CT team win probability using game state snapshots
"""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib
# matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def load_and_prepare_data(json_file="all_snapshots.json"):
    """Load snapshots and prepare features for ML training"""
    
    print(f"📊 Loading data from {json_file}...")
    with open(json_file, 'r') as f:
        snapshots = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(snapshots)
    
    print(f"✅ Loaded {len(df)} snapshots")
    print(f"📈 Data shape: {df.shape}")
    print(f"🏷️ Columns: {list(df.columns)}")
    
    # Create target variable (1 if CT wins, 0 if T wins)
    df['ct_wins'] = (df['winner'] == 'ct').astype(int)
    
    # Feature engineering
    df['player_advantage'] = df['cts_alive'] - df['ts_alive']
    df['ct_alive_ratio'] = df['cts_alive'] / (df['cts_alive'] + df['ts_alive'] + 1e-8)  # Avoid division by zero
    
    # Select features for training
    feature_columns = [
        'time_left',
        'cts_alive', 
        'ts_alive',
        'bomb_planted',
        'player_advantage',
        'ct_alive_ratio',
    ]
    
    X = df[feature_columns]
    y = df['ct_wins']
    
    print(f"🎯 Target distribution:")
    print(f"   CT wins: {y.sum()} ({y.mean():.1%})")
    print(f"   T wins:  {len(y) - y.sum()} ({1 - y.mean():.1%})")
    
    return X, y, feature_columns, df

def train_models(X, y):
    """Train multiple models and compare performance"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"🔄 Training set: {len(X_train)} samples")
    print(f"🧪 Test set: {len(X_test)} samples")
    
    # Scale features for logistic regression
    scaler = StandardScaler()
    # Fit with DataFrame to preserve feature names, then transform
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    
    # 1. Logistic Regression
    print("\n🔮 Training Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=2000, solver='liblinear')
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    models['logistic_regression'] = {
        'model': lr_model,
        'scaler': scaler,
        'predictions': lr_pred,
        'probabilities': lr_pred_proba,
        'accuracy': accuracy_score(y_test, lr_pred)
    }
    
    # 2. Random Forest
    print("🌲 Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    
    models['random_forest'] = {
        'model': rf_model,
        'scaler': None,
        'predictions': rf_pred,
        'probabilities': rf_pred_proba,
        'accuracy': accuracy_score(y_test, rf_pred)
    }
    
    # Print results
    print("\n📊 Model Performance:")
    for name, model_info in models.items():
        print(f"\n{name.replace('_', ' ').title()}:")
        print(f"  Accuracy: {model_info['accuracy']:.3f}")
        print("\n  Classification Report:")
        print(classification_report(y_test, model_info['predictions'], target_names=['T wins', 'CT wins']))
    
    return models, X_test, y_test

def analyze_feature_importance(models, feature_columns):
    """Analyze and visualize feature importance"""
    
    plt.figure(figsize=(12, 8))
    
    # Random Forest feature importance
    rf_importance = models['random_forest']['model'].feature_importances_
    
    plt.subplot(2, 1, 1)
    feature_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_importance
    }).sort_values('importance', ascending=True)
    
    plt.barh(feature_df['feature'], feature_df['importance'])
    plt.title('Random Forest - Feature Importance')
    plt.xlabel('Importance')
    
    # Logistic Regression coefficients
    lr_coef = models['logistic_regression']['model'].coef_[0]
    
    plt.subplot(2, 1, 2)
    coef_df = pd.DataFrame({
        'feature': feature_columns,
        'coefficient': np.abs(lr_coef)
    }).sort_values('coefficient', ascending=True)
    
    plt.barh(coef_df['feature'], coef_df['coefficient'])
    plt.title('Logistic Regression - Feature Coefficients (Absolute)')
    plt.xlabel('|Coefficient|')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print("📊 Feature importance plot saved as 'feature_importance.png'")
    
    print("\n🎯 Top 3 Most Important Features (Random Forest):")
    for i, (feature, importance) in enumerate(feature_df.tail(3)[['feature', 'importance']].values):
        print(f"  {i+1}. {feature}: {importance:.3f}")

def visualize_predictions(models, X_test, y_test):
    """Create visualizations of model predictions"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for i, (name, model_info) in enumerate(models.items()):
        row = i // 2
        col = i % 2
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, model_info['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['T wins', 'CT wins'], 
                   yticklabels=['T wins', 'CT wins'], ax=axes[row, col])
        axes[row, col].set_title(f'{name.replace("_", " ").title()} - Confusion Matrix')
        axes[row, col].set_ylabel('True Label')
        axes[row, col].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print("📊 Model performance plot saved as 'model_performance.png'")

def save_best_model(models, feature_columns):
    """Save the best performing model"""
    
    # Find best model by accuracy
    best_model_name = max(models.keys(), key=lambda k: models[k]['accuracy'])
    best_model_info = models[best_model_name]
    
    print(f"\n💾 Saving best model: {best_model_name} (Accuracy: {best_model_info['accuracy']:.3f})")
    
    # Save model and associated data
    model_data = {
        'model': best_model_info['model'],
        'scaler': best_model_info['scaler'],
        'feature_columns': feature_columns,
        'model_type': best_model_name,
        'accuracy': best_model_info['accuracy']
    }
    
    joblib.dump(model_data, 'ct_win_probability_model.pkl')
    print("✅ Model saved as 'ct_win_probability_model.pkl'")
    
    return best_model_name, best_model_info

def predict_win_probability(time_left, cts_alive, ts_alive, bomb_planted, model_file='ct_win_probability_model.pkl'):
    """Use trained model to predict CT win probability for a given game state"""
    
    # Load model
    model_data = joblib.load(model_file)
    model = model_data['model']
    scaler = model_data['scaler']
    feature_columns = model_data['feature_columns']
    
    # Create feature vector
    player_advantage = cts_alive - ts_alive
    ct_alive_ratio = cts_alive / (cts_alive + ts_alive + 1e-8)
    
    # Create DataFrame with proper feature names to avoid sklearn warning
    feature_data = {
        'time_left': time_left,
        'cts_alive': cts_alive, 
        'ts_alive': ts_alive, 
        'bomb_planted': bomb_planted,
        'player_advantage': player_advantage, 
        'ct_alive_ratio': ct_alive_ratio, 
    }
    
    X = pd.DataFrame([feature_data], columns=feature_columns)
    
    # Scale if needed
    if scaler is not None:
        X = scaler.transform(X)
    
    # Predict probability
    ct_win_prob = model.predict_proba(X)[0, 1]
    
    return ct_win_prob

def main():
    """Main training pipeline"""
    
    try:
        print("🎮 CS2 Win Probability Model Training")
        print("=" * 50)
        
        # Load and prepare data
        X, y, feature_columns, df = load_and_prepare_data()
        
        # Train models
        models, X_test, y_test = train_models(X, y)
        
        # Analyze feature importance
        analyze_feature_importance(models, feature_columns)
        
        # Visualize results
        visualize_predictions(models, X_test, y_test)
        
        # Save best model
        best_model_name, best_model_info = save_best_model(models, feature_columns)
        
        # Test prediction function
        print("\n🧪 Testing prediction function:")
        test_scenarios = [
            (60, 5, 5, False, "Equal teams, mid-round"),
            (30, 3, 5, False, "CT disadvantage"),
            (20, 5, 2, True, "CT advantage with bomb planted"),
            (5, 1, 1, True, "1v1 clutch with bomb planted"),
            (45, 4, 4, False, "Equal teams, early round"),
            (10, 2, 3, True, "CT slight disadvantage, bomb planted"),
            (90, 5, 3, False, "CT advantage, plenty of time"),
            (15, 1, 3, False, "CT major disadvantage, no bomb"),
            (3, 2, 1, True, "CT advantage, very low time, bomb planted"),
            (75, 5, 1, False, "CT major advantage, early round"),
            (25, 3, 3, True, "Equal teams, bomb planted, mid-time"),
            (8, 1, 2, False, "1v2 clutch for CT, no bomb"),
            (35, 4, 5, False, "T slight advantage, mid-round"),
            (12, 3, 1, True, "CT advantage, bomb planted, low time"),
            (50, 2, 4, False, "T major advantage, no bomb")
        ]
        
        for time_left, cts, ts, bomb, description in test_scenarios:
            prob = predict_win_probability(time_left, cts, ts, bomb)
            print(f"  {description}: {prob:.1%} CT win probability")
        
        print(f"\n✅ Training complete! Best model: {best_model_name}")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
