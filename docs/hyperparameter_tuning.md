# Hyperparameter Tuning for CS2 Win Probability Models

This document explains how to use the hyperparameter tuning functionality to optimize model performance automatically.

## Overview

The hyperparameter tuning system provides automated optimization for all supported model types:

- **XGBoost** models (including xgboost_hltv, xgboost_hltv_time)
- **LightGBM**
- **Random Forest**
- **Logistic Regression**
- **Neural Network (MLP)**

## Quick Start

### 1. Enable Hyperparameter Tuning

Edit `src/ml/train_win_probability_model.py` and modify the configuration:

```python
HYPERPARAMETER_TUNING = {
    'enabled': True,                # Enable tuning
    'models_to_tune': {
        'xgboost_hltv': True,       # Enable tuning for this model
        'lightgbm': True,           # Enable tuning for this model
        # ... enable other models as needed
    },
    'search_method': 'random',      # 'grid' or 'random'
    'search_intensity': 'quick',    # 'quick' or 'thorough'
    'n_iter': 20,                   # Iterations for random search
    'scoring': 'roc_auc',           # Optimization metric
    'cv_folds': 5,                  # Cross-validation folds
    'save_results': True            # Save results to file
}
```

### 2. Run Training

```bash
python src/ml/train_win_probability_model.py
```

## Configuration Options

### Search Methods

- **`'random'`**: Random search - faster, good coverage of parameter space
- **`'grid'`**: Grid search - exhaustive but slower

### Search Intensity

- **`'quick'`**: Smaller parameter space, faster execution
- **`'thorough'`**: Comprehensive parameter space, better results but slower

### Scoring Metrics

- **`'roc_auc'`**: Area under ROC curve (default, good for imbalanced data)
- **`'accuracy'`**: Classification accuracy
- **`'neg_log_loss'`**: Negative log loss (for probability calibration)

## Expected Runtime

Estimated tuning times for different configurations:

| Model               | Intensity | Search | Iterations | Estimated Time |
| ------------------- | --------- | ------ | ---------- | -------------- |
| XGBoost             | Quick     | Random | 20         | ~42 minutes    |
| LightGBM            | Quick     | Random | 20         | ~30 minutes    |
| Random Forest       | Quick     | Random | 20         | ~20 minutes    |
| Logistic Regression | Quick     | Random | 20         | ~4 minutes     |
| Neural Network      | Quick     | Random | 20         | ~60 minutes    |

_Times are estimates based on 5-fold cross-validation and may vary with dataset size_

## Parameter Spaces

### XGBoost (Quick)

```python
{
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.15],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1],
    'min_child_weight': [1, 3]
}
```

### LightGBM (Quick)

```python
{
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, -1],
    'learning_rate': [0.05, 0.1, 0.15],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'min_child_samples': [10, 20],
    'num_leaves': [31, 63]
}
```

_Thorough configurations include additional parameters and ranges_

## Output and Results

### Console Output

```
🎯 Hyperparameter tuning enabled for xgboost_hltv
📊 Estimated tuning time: 42m (17 combinations)
🔍 Starting XGBoost random search with 7 parameters...
   Total combinations: 17 (Random Search)
✅ Tuning complete! Best roc_auc: 0.8523
🏆 Best parameters: {'n_estimators': 200, 'max_depth': 5, ...}
💾 Hyperparameter tuning results saved to: outputs/hyperparameter_tuning/xgboost_hltv_hyperparameter_tuning_results.json
```

### Saved Results

Results are automatically saved to `outputs/hyperparameter_tuning/` with details including:

- Best parameters found
- Best cross-validation score
- Search configuration used
- Timestamp

## Recommendations

### For Development/Testing

```python
HYPERPARAMETER_TUNING = {
    'enabled': True,
    'models_to_tune': {'xgboost_hltv': True},  # One model only
    'search_method': 'random',
    'search_intensity': 'quick',
    'n_iter': 10,                              # Low iterations
    'cv_folds': 3,                             # Fewer folds
}
```

### For Production

```python
HYPERPARAMETER_TUNING = {
    'enabled': True,
    'models_to_tune': {
        'xgboost_hltv': True,
        'xgboost_hltv_time': True,
        'lightgbm': True
    },
    'search_method': 'random',
    'search_intensity': 'thorough',
    'n_iter': 100,                             # More iterations
    'cv_folds': 5,
}
```

## Advanced Usage

### Custom Parameter Grids

You can modify parameter grids in `src/ml/feature_sets.py`:

```python
HYPERPARAMETER_GRIDS['xgboost_custom'] = {
    'n_estimators': [100, 200, 500],
    'max_depth': [4, 6, 8],
    # ... your custom parameters
}
```

### Programmatic Access

```python
from src.ml.hyperparameter_tuning import HyperparameterTuner

tuner = HyperparameterTuner(scoring='roc_auc', cv_folds=5)
best_model, best_params, best_score = tuner.tune_xgboost(X_train, y_train)
```

## Troubleshooting

### Common Issues

1. **"No hyperparameter grid found"**

   - Check that the model name matches available configurations
   - Verify the model is in `feature_sets.HYPERPARAMETER_GRIDS`

2. **Long execution times**

   - Reduce `n_iter` for random search
   - Use `'quick'` intensity instead of `'thorough'`
   - Reduce `cv_folds` to 3

3. **Memory issues**
   - Enable tuning for one model at a time
   - Reduce dataset size for testing
   - Use smaller parameter grids

### Performance Tips

- **Start small**: Test with `n_iter=5-10` first
- **Use random search**: Usually better than grid search
- **Monitor resources**: Check CPU/memory usage during tuning
- **Incremental approach**: Tune one model type at a time

## Integration with Existing Workflow

The hyperparameter tuning integrates seamlessly with the existing training pipeline:

1. **If tuning is disabled**: Models use default parameters as before
2. **If tuning is enabled**: Best parameters replace defaults automatically
3. **Model saving**: Tuned models are saved with their optimized parameters
4. **Feature importance**: Works normally with tuned models
5. **Ensemble**: Can combine tuned and non-tuned models

## Files Created/Modified

- `src/ml/hyperparameter_tuning.py` - Main tuning module
- `src/ml/feature_sets.py` - Parameter grids and configurations
- `src/ml/train_win_probability_model.py` - Integrated tuning into training
- `outputs/hyperparameter_tuning/` - Results storage directory

This completes the hyperparameter tuning implementation for the CS2 win probability models.
