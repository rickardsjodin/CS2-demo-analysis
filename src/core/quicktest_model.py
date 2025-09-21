from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from src.ml.test_win_probability_scenarios import load_all_trained_models
import pandas as pd
from src.ml.feature_engineering import create_features


def predict(model_data, snapshot_data):
    df = pd.DataFrame([snapshot_data])
    try:
        df = create_features(df)
    except:
        print()

    # Ensure all required feature columns are present
    feature_columns = model_data['feature_columns']
    X = df[feature_columns]

    # Predict probability - handle both calibrated and original models
    model = model_data['model']
    ct_win_prob = model.predict_proba(X)[0, 1]
    return ct_win_prob


if __name__ == '__main__':
    all_models = load_all_trained_models()
    pred_model = all_models['xgboost']['data']

    sample_data = {
        'source': 'Round 8 in NiKo',
        'time_left': 1,
        'cts_alive': 3,
        'ts_alive': 3,
        'hp_ct': 201,
        'hp_t': 161,
        'bomb_planted': False,
        'ct_main_weapons': 3,
        't_main_weapons': 3,
        'ct_helmets': 3,
        't_helmets': 2,
        'defusers': 3,
        "ct_smokes":0,
        "ct_flashes":0,
        "ct_he_nades":0,
        "ct_molotovs":0,
        "ct_armor":3,
        "t_smokes":0,
        "t_flashes":0,
        "t_he_nades":0,
        "t_molotovs":0,
        "t_armor":3
        }
        

    res = float(predict(pred_model, sample_data))

    print(res)